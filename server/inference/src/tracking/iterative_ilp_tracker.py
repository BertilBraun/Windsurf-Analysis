from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import math
import numpy as np

from server.inference.src.visualization.stabilize import Transform
from ..util.video_io import VideoInfo, VideoReader
from ..common_types import Detection, Track
from ..settings import MAX_OVERLAP_LENGTH_SECONDS, OPTIMIZER_W_START
from .ILP_graph_solver import FragmentGraph, ILPGraphSolver
from server.inference.bot_sort.kalman_filter import KalmanFilter

EPS = 1e-9


# ───────────────────────── helpers: probabilities/metrics ───────────────────────── #


def chi2_dist(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    num = (p - q) ** 2
    den = p + q + eps
    return 0.5 * float((num / den).sum())


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def platt_prob_from_dist(d: float, a: float, b: float) -> float:
    # smaller distance → larger probability
    return sigmoid(a * (-d) + b)


def clamp_prob(p: float) -> float:
    return max(EPS, min(1.0 - EPS, float(p)))


def NLL_from_prob(p: float) -> float:
    """Negative log-likelihood ratio cost: -logit(p)."""
    p = clamp_prob(p)
    return float(-math.log(p / (1.0 - p)))


# ───────────────────────────── tracker implementation ───────────────────────────── #


@dataclass
class _KFCacheEntry:
    mean_end: np.ndarray  # (8,)
    cov_end: np.ndarray  # (8,8)
    end_frame: int


class IterativeILPTracker:
    """Iterative ILP tracker.

    Plan:
    1. Build a graph of possible fragment connections with their costs.
        - Possible connections (A -> B) are based on:
            - A.end_frame < B.start_frame and B.start_frame <= A.end_frame + MAX_OVERLAP_LENGTH_SECONDS * video_fps
        - Costs are based on:
            To calculate the actual cost, we use the sum of the NLL for motion, appearance and gap.
            - Motion: KF Mahalanobis NLL + GMC (position-only) + 0.5*log|S_pos|. We apply the appropriate Camera Transforms on each frame. Transfors are defined as: Transform = NamedTuple('Transform', [('dx', float), ('dy', float), ('da', float), ('frame_idx', int)]) # dx, dy, da for each frame relative to the previous frame.
            - Appearance: embedding is a LAB color histogram, we compute the mean histogram for both A and B and then use the chi-squared distance to get the appearance similarity probability by calculating platt_prob_from_dist. Lab χ² → Platt prob → NLLR
            - Gap: per-frame miss NLL
    2. Solve the ILP problem with a pretty low start_cost (no need to link up everything - it's fine to have some split tracks or even unassigned detections - it's iterative).
    3. Repeat from Step 1. but this time with the solution of the previous iteration as the starting point. We increase the start_cost by a small amount each time.
    4. Stop when the solution is stable (i.e. the cost of the solution is not changing much) or we have reached a maximum number of iterations (4 iterations).
    5. Return the solution.
    """

    def __init__(
        self,
        video_path: str,
        w_start: float = OPTIMIZER_W_START,
        # costs
        p_miss: float = 0.98,  # for gap NLL
        appearance_a: float = 7.427,  # Platt A
        appearance_b: float = 4.088,  # Platt B
        # motion eval
        motion_k_eval: int = 3,  # eval first K detections of B (1..3 recommended)
        d2_drop_threshold: float = 200.0,  # hard-drop average d^2 above this
        use_position_only: bool = True,  # gating_distance on (cx,cy) or (cx,cy,w,h)
        # iteration
        max_iters: int = 4,
        # optional internal long-gap split during iteration (simple rule)
        internal_split_gap_frames: int = 5,  # 0 disables; >0 splits tracks on internal gaps > this
        enable_edge_logging: bool = True,
    ) -> None:
        self.video_path = video_path
        self.w_start = float(w_start)
        self.p_miss = float(p_miss)
        self.appearance_a = float(appearance_a)
        self.appearance_b = float(appearance_b)
        self.motion_k_eval = int(max(1, motion_k_eval))
        self.d2_drop_threshold = float(d2_drop_threshold)
        self.use_position_only = bool(use_position_only)
        self.max_iters = int(max(1, max_iters))
        self.internal_split_gap_frames = int(max(0, internal_split_gap_frames))
        self.enable_edge_logging = bool(enable_edge_logging)

    # ───────────────────────────────── public API ───────────────────────────────── #

    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        """Run iterative graph building and ILP solve. Stops when cost does not improve."""
        # print tracks before tracking
        print('Tracks before tracking:')
        for track in tracks:
            print(
                f'Track {track.track_id}: {len(track.sorted_detections)} detections from {track.start_frame} to {track.end_frame}'
            )

        tracks = self._maybe_split_on_internal_gaps(tracks)
        best_cost = float('inf')

        print('Tracks after splitting on internal gaps:')
        for track in tracks:
            print(
                f'Track {track.track_id}: {len(track.sorted_detections)} detections from {track.start_frame} to {track.end_frame}'
            )

        frame_dict = {}
        with VideoReader(self.video_path) as reader:
            for frame_idx, frame in reader.read_frames():
                frame_dict[frame_idx] = frame

        for it in range(self.max_iters):
            graph = self._build_fragment_graph(tracks, video_properties.fps, transforms, frame_dict)
            # TODO increase start cost iteratively
            new_tracks, new_cost = ILPGraphSolver(self.w_start * (it + 1)).optimize_graph(graph)
            if new_cost >= best_cost:  # TODO smarter stopping condition (no assignment changes?)
                # no improvement → stop
                print(f'No improvement in iteration {it}, stopping. {new_cost} >= {best_cost}')
                break
            tracks, best_cost = new_tracks, new_cost
            # optional: after each solve, split again on internal gaps if enabled
            tracks = self._maybe_split_on_internal_gaps(tracks)

        return tracks

    # ─────────────────────────────── graph building ────────────────────────────── #

    def _build_fragment_graph(
        self, fragments: List[Track], video_fps: int, transforms: List[Transform], frame_dict: Dict[int, np.ndarray]
    ) -> FragmentGraph:
        """Build forward edges with costs. Logs one line per edge if enabled.
        If `enable_edge_logging` is True, also displays a debug visualization for each edge candidate
        right before adding the connection to the graph.
        """
        fragments = sorted(fragments, key=lambda t: t.start_frame)
        graph = FragmentGraph(fragments)

        # per-frame GMC mapping f → f+1: Transform
        transforms_dict: Dict[int, Transform] = {t.frame_idx: t for t in transforms}

        # KF cache: fit once per fragment (end state after all its detections)
        kf_cache = {i: self._fit_kf_end_state(frag, transforms_dict) for i, frag in enumerate(fragments)}

        max_gap_frames = int(round(video_fps * float(MAX_OVERLAP_LENGTH_SECONDS)))

        logs: List[str] = []

        N = len(fragments)
        for i in range(N):
            A = fragments[i]
            for j in range(i + 1, N):
                B = fragments[j]

                # forward-only, no same-frame overlap
                gap_frames = B.start_frame - A.end_frame - 1
                if gap_frames < 0 or gap_frames > max_gap_frames:
                    continue

                # compute costs
                motion_nll, avg_d2, used_k = self._motion_nll_cached(i, j, fragments, kf_cache, transforms_dict)
                if math.isinf(motion_nll) or math.isnan(motion_nll):
                    continue
                # hard-drop by average d^2 if requested
                if avg_d2 is not None and avg_d2 > self.d2_drop_threshold:
                    continue

                appearance_nll = self._appearance_nll(A, B)
                if math.isinf(appearance_nll) or math.isnan(appearance_nll):
                    continue

                gap_nll = self._gap_nll(gap_frames)

                total = motion_nll + appearance_nll + gap_nll

                # Debug visualization: show A.end frame and B.start frame with bboxes and cost terms
                if self.enable_edge_logging:
                    try:
                        cache_A = kf_cache[i]
                        self._show_edge_debug(
                            A,
                            B,
                            motion_nll,
                            appearance_nll,
                            gap_nll,
                            total,
                            frame_dict,
                            cache_A.mean_end,
                            cache_A.cov_end,
                            cache_A.end_frame,
                            transforms_dict,
                        )
                    except Exception:
                        pass

                graph.add_connection(i, j, float(total))

                if self.enable_edge_logging:
                    logs.append(
                        f'edge A[{i} id={A.track_id} end={A.end_frame}] -> '
                        f'B[{j} id={B.track_id} start={B.start_frame}] | '
                        f'Δ={gap_frames} | '
                        f'avg_d2={avg_d2:.3f} K={used_k} | '
                        f'C_mot={motion_nll:.4f} C_app={appearance_nll:.4f} C_gap={gap_nll:.4f} | '
                        f'C_tot={total:.4f}'
                    )

        if self.enable_edge_logging and logs:
            print('\n'.join(logs))

        return graph

    # ─────────────────────────────── debug visualization ─────────────────────────── #

    def _show_edge_debug(
        self,
        A: Track,
        B: Track,
        motion_nll: float,
        appearance_nll: float,
        gap_nll: float,
        total_cost: float,
        frame_dict: Dict[int, np.ndarray],
        kf_mean_end: np.ndarray,
        kf_cov_end: np.ndarray,
        kf_end_frame: int,
        transforms: Dict[int, Transform],
    ) -> None:
        try:
            import cv2  # type: ignore
        except Exception:
            return

        try:
            frame_a = frame_dict.get(A.end_frame)
            frame_b = frame_dict.get(B.start_frame)
            if frame_a is None or frame_b is None:
                return

            vis_a = frame_a.copy()
            vis_b = frame_b.copy()

            # Draw detection bounding boxes
            bb_a = A.end.bbox
            bb_b = B.start.bbox
            cv2.rectangle(vis_a, (int(bb_a.x1), int(bb_a.y1)), (int(bb_a.x2), int(bb_a.y2)), (0, 255, 0), 2)
            cv2.rectangle(vis_b, (int(bb_b.x1), int(bb_b.y1)), (int(bb_b.x2), int(bb_b.y2)), (0, 255, 255), 2)

            # Labels near boxes
            cv2.putText(
                vis_a,
                f'A id={A.track_id} end f={A.end_frame}',
                (max(0, int(bb_a.x1)), max(0, int(bb_a.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis_b,
                f'B id={B.track_id} start f={B.start_frame}',
                (max(0, int(bb_b.x1)), max(0, int(bb_b.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Prepare KF instance once for overlays
            kf = KalmanFilter()

            # Overlay KF current on A.end frame (mean_end, cov_end)
            try:
                cx_a, cy_a, w_a, h_a = kf.display_bbox(kf_mean_end, kf_cov_end, alpha=0.0)
                x1a = int(cx_a - w_a / 2.0)
                y1a = int(cy_a - h_a / 2.0)
                x2a = int(cx_a + w_a / 2.0)
                y2a = int(cy_a + h_a / 2.0)
                cv2.rectangle(vis_a, (x1a, y1a), (x2a, y2a), (255, 0, 0), 2)
                # Draw velocity arrow from KF state if available
                if kf_mean_end.shape[0] >= 6:
                    vx = float(kf_mean_end[4])
                    vy = float(kf_mean_end[5])
                    start_pt = (int(cx_a), int(cy_a))
                    end_pt = (int(round(cx_a + vx)), int(round(cy_a + vy)))
                    cv2.arrowedLine(vis_a, start_pt, end_pt, (255, 0, 0), 2, tipLength=0.3)
                cv2.putText(
                    vis_a,
                    'KF A end',
                    (x1a, max(0, y1a - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

            # Predict KF state from A.end to B.start and overlay on B.start frame
            try:
                m_b, P_b = kf.advance_state_to_frame(kf_mean_end, kf_cov_end, transforms, kf_end_frame, B.start_frame)
                cx_b, cy_b, w_b, h_b = kf.display_bbox(m_b, P_b, alpha=0.9)
                x1b = int(cx_b - w_b / 2.0)
                y1b = int(cy_b - h_b / 2.0)
                x2b = int(cx_b + w_b / 2.0)
                y2b = int(cy_b + h_b / 2.0)
                cv2.rectangle(vis_b, (x1b, y1b), (x2b, y2b), (255, 0, 255), 2)
                # Gating distance between predicted KF and B.start detection
                z = B.start.bbox.center_wh.reshape(1, 4).astype(np.float64)
                g2 = float(kf.gating_distance(m_b, P_b, z, only_position=self.use_position_only, metric='maha')[0])
                # Draw connection and label
                cv2.putText(
                    vis_b,
                    f'KF→B pred g^2={g2:.2f}',
                    (x1b, max(0, y1b - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                # Line between predicted KF center and detection center on B frame
                dcx = int((bb_b.x1 + bb_b.x2) / 2)
                dcy = int((bb_b.y1 + bb_b.y2) / 2)
                cv2.line(vis_b, (int(cx_b), int(cy_b)), (dcx, dcy), (0, 200, 255), 1)
                midx = int((cx_b + dcx) / 2)
                midy = int((cy_b + dcy) / 2)
                cv2.putText(
                    vis_b,
                    f'g^2={g2:.2f}',
                    (midx + 4, midy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

            # Normalize heights for side-by-side display
            ha, wa = vis_a.shape[:2]
            hb, wb = vis_b.shape[:2]
            target_h = max(ha, hb)
            scale_a = target_h / float(ha) if ha > 0 else 1.0
            scale_b = target_h / float(hb) if hb > 0 else 1.0
            vis_a_resized = cv2.resize(vis_a, (int(round(wa * scale_a)), target_h), interpolation=cv2.INTER_AREA)
            vis_b_resized = cv2.resize(vis_b, (int(round(wb * scale_b)), target_h), interpolation=cv2.INTER_AREA)
            combined = np.concatenate([vis_a_resized, vis_b_resized], axis=1)

            # Add a top banner with cost terms and gap frames
            banner_h = 36
            banner = np.full((banner_h, combined.shape[1], 3), 15, dtype=np.uint8)
            gap_frames = max(0, int(B.start_frame - A.end_frame - 1))
            text = (
                f'Δf={gap_frames}  '
                f'C_mot={motion_nll:.4f}  C_app={appearance_nll:.4f}  C_gap={gap_nll:.4f}  C_tot={total_cost:.4f}'
            )
            cv2.putText(banner, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            canvas = np.concatenate([banner, combined], axis=0)
            canvas = cv2.resize(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2))

            window_name = 'ILP edge debug (A end | B start)'
            cv2.imshow(window_name, canvas)
            # Short wait to refresh window without blocking the entire optimization
            cv2.waitKey(0)
        except Exception:
            # Fail silently in debug drawing
            return

    # ──────────────────────────────── cost helpers ─────────────────────────────── #

    def _appearance_nll(self, a: Track, b: Track) -> float:
        """Lab χ² distance between fragment prototypes → Platt → NLLR."""
        chi2 = chi2_dist(a.mean_embedding(), b.mean_embedding())
        p = platt_prob_from_dist(chi2, self.appearance_a, self.appearance_b)
        return NLL_from_prob(p)

    def _motion_nll_cached(
        self,
        idx_a: int,
        idx_b: int,
        frags: List[Track],
        kf_cache: Dict[int, _KFCacheEntry],
        transforms: Dict[int, Transform],
    ) -> Tuple[float, Optional[float], int]:
        """
        Motion NLL between frags[idx_a] → frags[idx_b]:
          - use cached KF end state for A
          - predict by Δ to each of first K detections of B
          - inverse-GMC the observation into A.end frame
          - 0.5*d2 + 0.5*log|S_pos|, averaged across used dets
        Returns: (motion_nll, avg_d2, used_K)
        """
        B = frags[idx_b]
        kf = KalmanFilter()

        cache = kf_cache[idx_a]
        m_end = cache.mean_end
        P_end = cache.cov_end
        t_end = cache.end_frame

        if not B.sorted_detections:
            return 0.0, 0.0, 0

        # evaluate up to K detections at B's start
        K = min(self.motion_k_eval, len(B.sorted_detections))
        total = 0.0
        d2_vals: List[float] = []
        used = 0

        m_pred, P_pred = m_end, P_end
        last_frame = t_end

        for k in range(K):
            db = B.sorted_detections[k]

            # predict from A.end by Δ
            m_pred, P_pred = kf.advance_state_to_frame(m_pred, P_pred, transforms, last_frame, db.frame_idx)
            last_frame = db.frame_idx

            z_obs_back = db.bbox.center_wh

            # position-only mahalanobis
            d2 = float(
                kf.gating_distance(
                    m_pred, P_pred, z_obs_back[None, :], only_position=self.use_position_only, metric='maha'
                )[0]
            )
            # add 0.5 * log|S_pos|
            _, S_full, _ = kf.project(m_pred, P_pred)
            S_pos = S_full[:2, :2] if self.use_position_only else S_full
            logdet = 0.5 * math.log(max(np.linalg.det(S_pos), EPS))

            total += 0.5 * d2 + logdet
            d2_vals.append(d2)
            used += 1

        if used == 0:
            return 1e6, None, 0

        avg_d2 = float(np.mean(d2_vals))
        motion_nll = total / used
        return motion_nll, avg_d2, used

    def _fit_kf_end_state(self, track: Track, transforms: Dict[int, Transform]) -> _KFCacheEntry:
        """Fit KF across all detections of a fragment and return its end state."""
        kf = KalmanFilter()

        first = track.start
        m, P = kf.initiate(first.bbox.center_wh)
        prev_f = track.start_frame

        for det in track.sorted_detections[1:]:
            gap = det.frame_idx - prev_f
            m, P = kf.predict(m, P, missed_frames=gap)
            m, P = kf.apply_forward_gmc_state(m, P, transforms[det.frame_idx])
            m, P = kf.update(m, P, det.bbox.center_wh)
            prev_f = det.frame_idx

        return _KFCacheEntry(mean_end=m, cov_end=P, end_frame=track.end_frame)

    # ───────────────────────────────────── gap ─────────────────────────────────── #

    def _gap_nll(self, gap_frames: int) -> float:
        return float(gap_frames) * (-math.log(self.p_miss))

    # ────────────────────────────── simple track splitter ───────────────────────── #

    def _maybe_split_on_internal_gaps(self, tracks: List[Track]) -> List[Track]:
        """
        Optional conservative splitter: if enabled, breaks tracks at internal gaps
        > `internal_split_gap_frames`. Keeps the same track_id for resulting fragments.
        If disabled, returns input unchanged.
        """
        G = self.internal_split_gap_frames
        if G <= 0:  # Skip splitting - disabled # TODO what about other parameters for KF or Appearance uncertainty?
            return tracks

        out: List[Track] = []
        for tr in tracks:
            if len(tr.sorted_detections) <= 1:
                out.append(tr)
                continue
            run: List[Detection] = []
            last_f = None
            for d in tr.sorted_detections:
                if last_f is None or (d.frame_idx - last_f) <= G:  # TODO also split on KF or Appearance uncertainty
                    run.append(d)
                else:
                    # TODO does that work with multiple tracks with the same track_id?
                    out.append(Track(track_id=tr.track_id, sorted_detections=run))
                    run = [d]
                last_f = d.frame_idx
            if run:
                out.append(Track(track_id=tr.track_id, sorted_detections=run))
        return out
