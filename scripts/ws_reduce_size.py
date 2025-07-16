#!/usr/bin/env python3
"""
windurf_video_shrinker.py

Batch shrink Canon RP (or any) MP4 action-sports clips to 2 practical distribution tiers:

  1. "send"  – aggressively small, e.g., <64 MiB per clip, good for messaging / email.
  2. "keep"  – high visual quality archival/share copy that is *much* smaller than camera originals.

Key ideas
---------
* Uses ffprobe to read duration / frame rate / resolution / bitrates.
* Computes bitrate budget for the "send" tier from desired *maximum total file size*.
* Optionally scales down resolution and/or frame rate for "send" (defaults: <=720p, <=30fps) to save bits.
* 2‑pass rate control for predictable target size in "send" tier; falls back gracefully for very short clips.
* "keep" tier encodes visually transparent-ish copy using CRF‑based encode (size floats with content) –
  you get a big size reduction versus camera originals without the headache of picking bitrates.
* Automatically prefers HEVC (libx265) when available for better compression; otherwise uses H.264 (libx264).
* Simple CLI: supply one or more input files. Optional -o output directory. Optional --profiles to choose which
  tiers to create. Reasonable defaults; override almost everything with flags.

Example usage
-------------
# Create _send (<=64MiB) and _keep versions alongside sources:
$ python windsurf_video_shrinker.py session/*.MP4

# Create only send copies, capped at 48MiB, outputs to ./shareable:
$ python windsurf_video_shrinker.py -p send -S 48 -o shareable session/*.MP4

# Create keep copies at 1080p max height and CRF 22 (x265 if available):
$ python windsurf_video_shrinker.py -p keep --keep-max-height 1080 --keep-crf 22 session/*.MP4

Installation prerequisites
-------------------------
* ffmpeg + ffprobe installed and on PATH.
  - macOS (homebrew):  brew install ffmpeg
  - Linux (apt):        sudo apt install ffmpeg
  - Windows (choco):    choco install ffmpeg

Return codes
------------
* 0 on overall success (all requested outputs produced or skipped because already within constraints).
* Non‑zero on first encountered fatal error.

"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

###############################################################################
# Utilities
###############################################################################

def run(cmd, *, quiet=False, check=True):
    """Run a subprocess command.

    Args:
        cmd (list[str] | str): Command + args.
        quiet (bool): If True, suppress stdout/stderr capture in terminal; still capture for return.
        check (bool): Raise CalledProcessError on non‑zero exit.
    Returns:
        CompletedProcess
    """
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = cmd
    if not quiet:
        print("→", " ".join(shlex.quote(c) for c in cmd_list))
    return subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check, text=True)


def which_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        sys.exit("ERROR: ffmpeg/ffprobe not found in PATH. Install ffmpeg before running.")
    return ffmpeg, ffprobe


###############################################################################
# ffprobe metadata helpers
###############################################################################

def ffprobe_info(ffprobe_bin, infile: Path):
    """Return dict with selected metadata for infile using ffprobe JSON output."""
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(infile),
    ]
    cp = run(cmd, quiet=True)
    data = json.loads(cp.stdout)

    fmt = data.get("format", {})
    streams = data.get("streams", [])

    duration = float(fmt.get("duration", 0.0)) if fmt.get("duration") else 0.0
    size_bytes = int(fmt.get("size", 0)) if fmt.get("size") else 0

    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    v = {}
    if vstream:
        v["codec"] = vstream.get("codec_name")
        v["width"] = int(vstream.get("width", 0))
        v["height"] = int(vstream.get("height", 0))
        # r_frame_rate like "60000/1001"
        rfr = vstream.get("r_frame_rate", "0/0")
        try:
            num, den = rfr.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except Exception:  # noqa: BLE001
            fps = 0.0
        v["fps"] = fps
        v["bit_rate"] = int(vstream.get("bit_rate", 0)) if vstream.get("bit_rate") else 0

    a = {}
    if astream:
        a["codec"] = astream.get("codec_name")
        a["channels"] = int(astream.get("channels", 0))
        a["sample_rate"] = int(astream.get("sample_rate", 0)) if astream.get("sample_rate") else 0
        a["bit_rate"] = int(astream.get("bit_rate", 0)) if astream.get("bit_rate") else 0

    return {
        "duration": duration,
        "size_bytes": size_bytes,
        "video": v,
        "audio": a,
    }


###############################################################################
# Bitrate math for size‑targeted encodes
###############################################################################

def compute_bitrates_for_target(total_mebibytes: float, duration_s: float, *, audio_kbps: int) -> int:
    """Compute *video* bitrate (kbps) to hit total size budget.

    total_bits = target_size_bytes * 8.
    subtract audio_bits = audio_kbps*1000 * duration_s.
    We leave 5% safety margin for container overhead + ratecontrol variance.
    Returns >=100 kbps minimum safeguard.
    """
    if duration_s <= 0:
        # fallback: just pick something modest
        return max(100, int((total_mebibytes * 1024 * 8 * 0.95)))  # treat total as video only

    total_bits = total_mebibytes * 1024 * 1024 * 8
    audio_bits = audio_kbps * 1000 * duration_s
    video_bits = max(0, total_bits - audio_bits)
    video_bits *= 0.95  # safety margin
    v_kbps = int(video_bits / duration_s / 1000)
    return max(100, v_kbps)


###############################################################################
# Encoding command builders
###############################################################################

def build_scale_filter(src_w, src_h, max_height=None, max_width=None):
    """Return ffmpeg -vf scale expression to *not upsize*; maintain aspect."""
    if max_height is None and max_width is None:
        return None
    # We'll enforce whichever constraint is tighter; use ffmpeg's force_original_aspect_ratio=decrease
    parts = []
    if max_height is not None and max_width is not None:
        # Use scale=min(max_width,iw) :-2 ensures mod2 even width/height.
        # Simpler: use scale=w=-2:h={max_height} with FOAR=decrease when both dims given.
        parts.append(f"scale=-2:{max_height}:force_original_aspect_ratio=decrease")
    elif max_height is not None:
        parts.append(f"scale=-2:{max_height}:")
    elif max_width is not None:
        parts.append(f"scale={max_width}:-2:")
    filt = ",".join(parts).rstrip(":")
    return filt or None


def build_fps_filter(limit_fps, src_fps):
    if limit_fps is None:
        return None
    if src_fps <= 0:
        return f"fps={limit_fps}"
    if src_fps <= limit_fps + 0.01:  # no need to drop
        return None
    return f"fps={limit_fps}"


def maybe_chain_filters(*filters):
    fs = [f for f in filters if f]
    if not fs:
        return None
    return ",".join(fs)


def detect_hevc_support(ffmpeg_bin):
    try:
        cp = run([ffmpeg_bin, "-hide_banner", "-encoders"], quiet=True, check=False)
    except Exception:  # noqa: BLE001
        return False
    return "libx265" in cp.stdout


###############################################################################
# Actual encode workers
###############################################################################

def encode_send(ffmpeg_bin, src: Path, dst: Path, meta: dict, *, size_mib: float, max_height: int, max_fps: float,
                vcodec: str = "libx264", preset: str = "slow", audio_kbps: int = 64, two_pass: bool = True,
                extra_vopts=None):
    """Encode a size‑targeted send copy."""
    if dst.exists():
        print(f"[send] Skipping {src.name}: destination exists.")
        return

    duration = meta.get("duration", 0.0) or 0.0
    src_v = meta.get("video", {})
    v_w, v_h, v_fps = src_v.get("width", 0), src_v.get("height", 0), src_v.get("fps", 0.0)

    scale = build_scale_filter(v_w, v_h, max_height=max_height)
    fps_f = build_fps_filter(max_fps, v_fps)
    vf = maybe_chain_filters(scale, fps_f)

    vk = compute_bitrates_for_target(size_mib, duration, audio_kbps=audio_kbps)
    maxrate = int(vk * 1.5)
    bufsize = int(vk * 3)

    # For *very* short clips the computed bitrate can explode; cap to something sane unless user overrides.
    if duration < 5 and vk > 10000:  # >10 Mbps for sub‑5s clip is wasteful for a tiny target
        vk = 5000
        maxrate = 7500
        bufsize = 15000

    common = [ffmpeg_bin, "-y", "-i", str(src)]
    if vf:
        common += ["-vf", vf]

    if two_pass:
        with tempfile.TemporaryDirectory(prefix="ffm_pass_", dir=str(dst.parent)) as td:
            passlog = str(Path(td) / "ffm2pass")
            # pass 1
            cmd1 = (
                common
                + [
                    "-c:v", vcodec,
                    "-b:v", f"{vk}k",
                    "-maxrate", f"{maxrate}k",
                    "-bufsize", f"{bufsize}k",
                    "-pass", "1",
                    "-passlogfile", passlog,
                    "-preset", preset,
                ]
            )
            if extra_vopts:
                cmd1 += extra_vopts
            # No audio in pass1 to speed up
            cmd1 += [
                "-an",
                "-f", "mp4",
                os.devnull,
            ]
            run(cmd1)

            # pass 2
            cmd2 = (
                common
                + [
                    "-c:v", vcodec,
                    "-b:v", f"{vk}k",
                    "-maxrate", f"{maxrate}k",
                    "-bufsize", f"{bufsize}k",
                    "-pass", "2",
                    "-passlogfile", passlog,
                    "-preset", preset,
                    "-c:a", "aac",
                    "-b:a", f"{audio_kbps}k",
                    "-movflags", "+faststart",
                    str(dst),
                ]
            )
            if extra_vopts:
                cmd2 += extra_vopts
            run(cmd2)
    else:
        # Single‑pass (less predictable size)
        cmd = (
            common
            + [
                "-c:v", vcodec,
                "-b:v", f"{vk}k",
                "-maxrate", f"{maxrate}k",
                "-bufsize", f"{bufsize}k",
                "-preset", preset,
                "-c:a", "aac",
                "-b:a", f"{audio_kbps}k",
                "-movflags", "+faststart",
                str(dst),
            ]
        )
        if extra_vopts:
            cmd += extra_vopts
        run(cmd)

    # Post‑check size
    if dst.exists():
        out_size = dst.stat().st_size / (1024 * 1024)
        if out_size > size_mib * 1.05:
            print(f"WARNING: {dst.name} is {out_size:.1f} MiB > target {size_mib} MiB.")
        else:
            print(f"[send] {dst.name}: {out_size:.1f} MiB (target {size_mib} MiB).")


def encode_keep(ffmpeg_bin, src: Path, dst: Path, meta: dict, *, max_height: int | None,
                vcodec_pref: str, crf: int, preset: str, tune: str | None,
                audio_kbps: int, copy_audio: bool = True, extra_vopts=None):
    """Encode a quality‑oriented keep copy (CRF‑based)."""
    if dst.exists():
        print(f"[keep] Skipping {src.name}: destination exists.")
        return

    src_v = meta.get("video", {})
    v_w, v_h = src_v.get("width", 0), src_v.get("height", 0)

    vf = None
    if max_height is not None and v_h > max_height:
        vf = build_scale_filter(v_w, v_h, max_height=max_height)

    cmd = [ffmpeg_bin, "-y", "-i", str(src)]
    if vf:
        cmd += ["-vf", vf]

    vcodec = vcodec_pref
    cmd += ["-c:v", vcodec, "-crf", str(crf), "-preset", preset]
    if tune:
        cmd += ["-tune", tune]
    if extra_vopts:
        cmd += extra_vopts

    if copy_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", f"{audio_kbps}k"]

    cmd += ["-movflags", "+faststart", str(dst)]

    run(cmd)

    if dst.exists():
        out_size = dst.stat().st_size / (1024 * 1024)
        print(f"[keep] {dst.name}: {out_size:.1f} MiB.")


###############################################################################
# Main CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(
        description="Shrink high‑res/high‑fps windsurfing MP4 videos into shareable + archival tiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="Input video files (MP4, MOV, etc.)")
    p.add_argument("-o", "--outdir", type=str, default=None, help="Output directory (created if needed). If omitted, outputs placed next to each input.")
    p.add_argument("-p", "--profiles", choices=["send", "keep"], default=["send", "keep"], help="Which output profiles to produce.")

    # send profile options
    p.add_argument("-S", "--send-size", type=float, default=64.0, help="Max MiB for 'send' outputs.")
    p.add_argument("--send-max-height", type=int, default=1080, help="Max height for 'send' (no upscale). Use 0 to keep original.")
    p.add_argument("--send-max-fps", type=float, default=30.0, help="Max frame rate for 'send'. Use 0 to keep original.")
    p.add_argument("--send-audio-kbps", type=int, default=64, help="AAC audio bitrate for 'send'.")
    p.add_argument("--send-no-2pass", action="store_true", help="Disable 2‑pass encode for 'send' (faster, less size accuracy).")

    # keep profile options
    p.add_argument("--keep-max-height", type=int, default=0, help="Max height for 'keep'; 0=keep original.")
    p.add_argument("--keep-crf", type=int, default=23, help="CRF quality for 'keep' (lower=better/larger). For libx264 typical 18‑23; for libx265 typical 20‑28.")
    p.add_argument("--keep-preset", type=str, default="slow", help="Encoder preset speed/efficiency (x264/x265 presets: ultrafast..veryslow).")
    p.add_argument("--keep-audio-kbps", type=int, default=160, help="Reencode audio bitrate if not copying.")
    p.add_argument("--keep-reencode-audio", action="store_true", help="Reencode audio instead of streamcopy.")
    p.add_argument("--keep-tune", type=str, default=None, help="Optional encoder tune (e.g., 'film', 'fastdecode', 'grain').")

    # codec selection
    p.add_argument("--force-x264", action="store_true", help="Force H.264 even if x265 available.")
    p.add_argument("--force-x265", action="store_true", help="Force HEVC; error if unsupported.")

    # misc
    p.add_argument("--dry-run", action="store_true", help="Print what would be done without encoding.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    return p.parse_args()


###############################################################################
# Path helpers
###############################################################################

def output_path(infile: Path, outdir: Path | None, suffix: str) -> Path:
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        base = infile.stem
        return outdir / f"{base}{suffix}.mp4"
    # same directory
    return infile.with_name(f"{infile.stem}{suffix}.mp4")


###############################################################################
# Workflow
###############################################################################

def process_file(ffmpeg_bin, ffprobe_bin, infile: Path, args, hevc_available: bool):
    if not infile.exists():
        print(f"WARNING: input not found: {infile}")
        return

    meta = ffprobe_info(ffprobe_bin, infile)
    dur = meta.get("duration", 0.0)
    in_size_mib = meta.get("size_bytes", 0) / (1024 * 1024)
    v = meta.get("video", {})
    fps = v.get("fps", 0.0)
    wh = f"{v.get('width','?')}x{v.get('height','?')}@{fps:.2f}fps"

    print(f"Input: {infile.name} | {in_size_mib:.1f} MiB | {dur:.1f}s | {wh}")

    # Decide codecs
    if args.force_x264 and args.force_x265:
        sys.exit("ERROR: --force-x264 and --force-x265 are mutually exclusive.")

    if args.force_x265:
        if not hevc_available:
            sys.exit("ERROR: --force-x265 specified but libx265 not available in this ffmpeg build.")
        send_vcodec = keep_vcodec = "libx265"
    elif args.force_x264:
        send_vcodec = keep_vcodec = "libx264"
    else:
        # heuristic: use x264 for send (compatibility), x265 for keep if available.
        send_vcodec = "libx264"
        keep_vcodec = "libx265" if hevc_available else "libx264"

    # PROFILES -----------------------------------------------------------------
    if "send" in args.profiles:
        if in_size_mib <= args.send_size * 1.01:
            print(f"[send] Skipping: already <= target ({in_size_mib:.1f} MiB). Use --overwrite to force.")
        else:
            dst = output_path(infile, Path(args.outdir) if args.outdir else None, f"_send{int(args.send_size)}MiB")
            if dst.exists() and not args.overwrite:
                print(f"[send] Exists: {dst} (use --overwrite).")
            else:
                if args.dry_run:
                    print(f"DRYRUN send -> {dst}")
                else:
                    encode_send(
                        ffmpeg_bin,
                        infile,
                        dst,
                        meta,
                        size_mib=args.send_size,
                        max_height=None if args.send_max_height <= 0 else args.send_max_height,
                        max_fps=None if args.send_max_fps <= 0 else args.send_max_fps,
                        vcodec=send_vcodec,
                        preset="slow",
                        audio_kbps=args.send_audio_kbps,
                        two_pass=not args.send_no_2pass,
                    )

    if "keep" in args.profiles:
        dst = output_path(infile, Path(args.outdir) if args.outdir else None, "_keep")
        if dst.exists() and not args.overwrite:
            print(f"[keep] Exists: {dst} (use --overwrite).")
        else:
            if args.dry_run:
                print(f"DRYRUN keep -> {dst}")
            else:
                encode_keep(
                    ffmpeg_bin,
                    infile,
                    dst,
                    meta,
                    max_height=None if args.keep_max_height <= 0 else args.keep_max_height,
                    vcodec_pref=keep_vcodec,
                    crf=args.keep_crf,
                    preset=args.keep_preset,
                    tune=args.keep_tune,
                    audio_kbps=args.keep_audio_kbps,
                    copy_audio=not args.keep_reencode_audio,
                )


###############################################################################
# Entrypoint
###############################################################################

def main():
    args = parse_args()
    ffmpeg_bin, ffprobe_bin = which_ffmpeg()
    hevc_available = detect_hevc_support(ffmpeg_bin)

    if args.outdir is not None:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for inp in args.inputs:
        process_file(ffmpeg_bin, ffprobe_bin, Path(inp), args, hevc_available)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
