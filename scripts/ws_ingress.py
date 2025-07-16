#!/usr/bin/env python3
import argparse, os, queue, subprocess, threading, time
from datetime import datetime
from pathlib import Path
import shutil
import logging
import shlex
from collections import defaultdict
import sys
import signal
from util import stabilize_ffmpeg, setup_logging, MpvIPCTask, get_tmp_file
# TODO: remove tqdm dependency
from tqdm import tqdm

MARK_LUA_SCRIPT = Path(__file__).parent / "mpv_script_cutting.lua"

def signal_handler(_sig, _frame):
    logging.warning("Interrupted! Shutting down workers...")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi"}

g_apply_stabilized = True
g_output_pre_stabilized = False

def tagify_str(s: str) -> str:
    return s.strip().replace("\n", "_").replace("\r", "").replace(" ", "_")

def ask_tags_popup(default="") -> str:
    try:
        res = subprocess.run(
            ["zenity", "--entry", "--text=Enter tags:", "--entry-text", default],
            capture_output=True, text=True)
        if res.returncode == 0:
            return tagify_str(res.stdout.strip())
        else:
            return ""
    except FileNotFoundError:
        logging.warning("Zenity not installed, falling back to input().")
        return tagify_str(input("Enter tags: "))

def rotation_vf_string(angle):
    # Returns None if no rotation needed, else the correct -vf string for ffmpeg
    if angle == 0 or angle is None:
        return None
    elif angle == 90:
        return "transpose=1"
    elif angle == 180:
        return "transpose=1,transpose=1"
    elif angle == 270:
        return "transpose=2"
    else:
        logging.warning(f"Unknown rotation angle: {angle}, ignoring.")
        return None

def cut_worker_inner(item, done_list):
    seg, src, dst = item
    print("cutting:", item)
    angle = seg.get("rotate", 0)

    do_cut = not (seg["end"] is None and seg["start"] == 0 and seg["rotate"] == 0)
    use_temp_file = do_cut
    if use_temp_file:
        cut_file = get_tmp_file(suffix=src.suffix)
    else:
        cut_file = src

    do_stabilize = seg.get("stabilize", g_apply_stabilized)

    try:
        if do_cut:
            cmd_cut = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", str(seg["start"])
            ]

            if seg["end"] is not None:
                cmd_cut += ["-to", str(seg["end"])]
            cmd_cut += ["-i", str(src)]

            vf = rotation_vf_string(angle)
            if vf:
                cmd_cut += ["-vf", vf]
                cmd_cut += ["-c:v", "libx264"]
            cmd_cut += ["-c:a", "copy", cut_file, "-y"]

            logging.info(f"  ✂ cutting: {shlex.join(cmd_cut)}")
            try:
                subprocess.run(cmd_cut, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"  !! ffmpeg failed for cut {dst}: {e}")
                return

        basename = dst.with_suffix('').name
        if g_output_pre_stabilized or not do_stabilize:
            raw_clip = dst.with_name(f"{basename}{dst.suffix}")
            logging.info(f"  📋 copying pre-stabilized to {raw_clip}")
            try:
                shutil.copy(cut_file, raw_clip)
            except Exception as e:
                logging.error(f"  !! failed to copy pre-stabilized file: {e}")
            done_list.append(raw_clip)

        # Optionally stabilize
        if do_stabilize:
            stabilized_path = dst.with_name(f"{basename}_stabilized{dst.suffix}")
            logging.info(f"  📈 stabilizing: {stabilized_path}")
            success = stabilize_ffmpeg(cut_file, stabilized_path)
            if success:
                done_list.append(stabilized_path)
            else:
                raw_clip = dst.with_name(f"{basename}{dst.suffix}")
                logging.warning(f"  !! stabilization failed, copying _no_stabilize to output ({raw_clip})")
                if not g_output_pre_stabilized:
                    try:
                        shutil.copy(cut_file, raw_clip)
                    except Exception as e:
                        logging.info(f"  !! failed to copy fallback raw clip: {e}")
                done_list.append(raw_clip)
        else:
            logging.info("  ➡ skipping stabilization.")

    finally:
        if use_temp_file:
            try:
                os.remove(cut_file)
            except Exception:
                pass

def cut_worker(task_queue, done_list):
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.task_done()
            break
        try:
            cut_worker_inner(item, done_list)
        except Exception as e:
            logging.error(f"  !! Error in cut_worker: {e}")
        finally:
            task_queue.task_done()

filename_counter = defaultdict(int)

def generate_video_path(base_name: str, out_dir: Path, in_video: Path) -> Path:
    date_prefix = datetime.fromtimestamp(
        in_video.stat().st_mtime).strftime("%Y_%m_%d")
    base_name = f"{date_prefix}_{base_name}"
    filename_counter[base_name] += 1
    uid = f"{filename_counter[base_name]:03d}"
    dst = out_dir / f"{base_name}_{uid}{in_video.suffix}" if uid else out_dir / f"{base_name}{in_video.suffix}"
    while dst.exists():
        filename_counter[base_name] += 1
        uid = f"{filename_counter[base_name]:03d}"
        dst = out_dir / f"{base_name}_{uid}{in_video.suffix}"
    return dst

def annotate_and_cut(input_path: Path, out_dir: Path, n_workers: int, no_input: bool = False):
    if input_path.is_file():
        videos = [input_path]
    else:
        videos = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXT])
    out_dir.mkdir(parents=True, exist_ok=True)
    segment_queue = queue.Queue()
    n_tasks = 0
    done_segments = []

    workers = [
        threading.Thread(target=cut_worker, args=(segment_queue, done_segments), daemon=True)
        for _ in range(n_workers)
    ]
    for w in workers: w.start()

    quit = False
    for i, vid in enumerate(videos):
        do_stabilize = g_apply_stabilized
        logging.info(f"\n=== {vid.relative_to(input_path)} ===")
        if no_input:
            dst = generate_video_path(vid.stem, out_dir, vid)
            seg = {
                "source": str(vid),
                "start": 0,
                "end": None,
                "rotate": 0,
                "stabilize": g_apply_stabilized
            }
            segment_queue.put((seg, vid, dst))
            n_tasks += 1
            logging.info("    ✓ segment queued: %s", dst.name)
            continue

        try:
            with MpvIPCTask(vid, MARK_LUA_SCRIPT) as mpv:
                current_start = None
                mpv.show_text(f"[{i}/{len(videos)}] processed", 5)
                while mpv.is_running() and not quit:
                    events = mpv.poll_events()
                    for ev in events:
                        if ev.get("event") == "client-message":
                            args = ev.get("args", [])
                            if not args:
                                continue
                            msg = args[0]
                            if msg == "m_mark_start":
                                current_start = mpv.get_property("time-pos")
                                mpv.show_text(f"Mark start: {current_start}", 3)
                                logging.info(f"  ▶ start @ {current_start:.3f}s")
                            elif msg == "m_mark_end" and current_start is not None:
                                end_time = mpv.get_property("time-pos")
                                logging.info(f"  ■ end   @ {end_time:.3f}s")

                                rotate = mpv.get_property("video-rotate")
                                rotate = int(rotate or 0)

                                mpv.show_text("Enter tags in popup", 3)
                                tags = ask_tags_popup()

                                dst = generate_video_path(tags, out_dir, vid)
                                seg = {
                                    "source": str(vid),
                                    "start": current_start,
                                    "end": end_time,
                                    "rotate": rotate,
                                    "stabilize": do_stabilize
                                }
                                segment_queue.put((seg, vid, dst))
                                n_tasks += 1
                                logging.info(f"    ✓ segment queued: {dst.name}")
                                current_start = None
                            elif msg == "m_keep":
                                logging.info("  ➡ keeping the whole video")
                                mpv.show_text("Keeping the whole video", 3)
                                tags = ask_tags_popup(vid.stem)
                                dst = generate_video_path(tags, out_dir, vid)
                                seg = {
                                    "source": str(vid),
                                    "start": 0,
                                    "end": None,
                                    "rotate": 0,
                                    "stabilize": do_stabilize
                                }
                                segment_queue.put((seg, vid, dst))
                                n_tasks += 1
                                logging.info(f"    ✓ segment queued: {dst.name}")
                            elif msg == "m_quit":
                                logging.info(" ➡ quitting video cutting")
                                mpv.show_text("Quitting video cutting", 3)
                                quit = True
                            elif msg == "m_toggle_stabilize":
                                do_stabilize = not do_stabilize
                                status = "enabled" if do_stabilize else "disabled"
                                logging.info(f"  ➡ stabilization {status}")
                                mpv.show_text(f"Stabilization {status}", 3)
                            else:
                                logging.warning(f"  !! unknown message: {msg}")

        except Exception as e:
            logging.error(f"!! mpv error for {vid}: {e}")

        if quit:
            break

    # Signal workers to finish and wait
    for _ in workers: segment_queue.put(None)

    with tqdm(total=n_tasks, desc="Processing segments", unit="segment") as pbar:
        while not segment_queue.all_tasks_done:
            pbar.update(n_tasks - segment_queue.unfinished_tasks)
            time.sleep(0.5)

    segment_queue.join()
    logging.info(f"\nAll done! {len(done_segments)} clips written to {out_dir}")

def main():
    global g_apply_stabilized, g_output_pre_stabilized

    setup_logging()

    parser = argparse.ArgumentParser(description="Interactive windsurfing video annotator & cutter")
    parser.add_argument("input_path", type=Path, help="Folder with videos to annotate or single video file.")
    parser.add_argument("output_dir", type=Path, help="Where to write cut clips")
    parser.add_argument("--workers", type=int, default=2, help="Parallel ffmpeg workers")
    parser.add_argument("--stabilize", action="store_true", help="Stabilize clips using ffmpeg vidstab")
    parser.add_argument("--keep-cut", action="store_true", help="Output pre-stabilized (raw cut) clip with _no_stabilize suffix even if stabilization is applied.")
    parser.add_argument("--no-input", action="store_true", help="Skip annotation and mpv; just process/copy/stabilize all video files.")

    args = parser.parse_args()
    g_apply_stabilized = args.stabilize
    g_output_pre_stabilized = args.keep_cut
    annotate_and_cut(args.input_path, args.output_dir, args.workers, no_input=args.no_input)

if __name__ == "__main__":
    main()
