#!/usr/bin/env python3
"""
Branch-and-Bound (Backtracking) Fragment Linker for Multi-Object Tracking (MOT)
-----------------------------------------------------------------------------
This module implements a global branch & bound search over *fragments* (short
pre-linked detection sequences) to assemble full trajectories while minimizing
an additive cost functional:

    TotalCost = Σ_tracks ( w_birth
                           + Σ_edges_in_track [ w_iou*(1 - IoU)
                                               + w_app*(1 - cosine_sim)
                                               + w_gap*gap_norm ] )

Key Constraints
---------------
* Each fragment can have at most one predecessor and one successor.
* Temporal ordering must be respected (successor starts strictly after
  predecessor ends, with optional gap).
* A candidate edge (fragment A -> B) is feasible only if the temporal gap is
  ≤ max_stitch_gap_frames.

Branch & Bound Strategy
-----------------------
At each step we decide for the next (ordered) fragment either to start a new
track (pay birth cost) or link it to a feasible predecessor (pay edge cost).
We maintain a running best solution (upper bound) and prune subtrees whose
current partial cost plus a *lower bound* on remaining unassigned fragments
exceeds this best.

Lower Bound (Conservative)
--------------------------
For every unassigned fragment f (with no predecessor fixed yet), we add:
    LB_f = min( w_birth, best_incoming_edge_cost(f) )
Summing these yields a valid optimistic lower bound because future global
structural / cohesion penalties are ignored (optimistic) and each fragment must
incur at least one of those two costs (either starting a track, or being linked
via some edge not cheaper than its best possible incoming edge).

Heuristic Ordering
------------------
Fragments are expanded in an order that tends to tighten bounds faster: by
(default) ascending (best_incoming - w_birth) and then by start frame. This
promotes early resolution of fragments that strongly prefer linkage over birth.

Usage
-----
tracker = BranchAndBoundFragmentTracker()
tracks = tracker.track_detections(detections, video_info)

Set `use_preprocessor=False` to treat every detection as a singleton fragment.
Set `return_debug=True` to also receive internal linkage arrays.

Notes / Extensions
------------------
The core implementation here focuses on the original local edge cost model.
Additional global cohesion metrics (appearance dispersion, motion smoothness,
margin ambiguity penalties, etc.) can be integrated by converting their
contributions into incremental edge or append deltas and *excluding* them from
the lower bound for conservativeness.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# External project imports (assumed to exist in user's codebase)
from video_io import VideoInfo
from common_types import Detection, FrameIndex, Track, TrackId, cosine_similarity
from greedy_preprocessor import GreedyPreprocessor


# ---------------------------------------------------------------------------
# Internal Fragment Metadata
# ---------------------------------------------------------------------------
@dataclass
class _FragMeta:
    idx: int
    start_frame: int
    end_frame: int
    start_det: Detection
    end_det: Detection


# ---------------------------------------------------------------------------
# Branch & Bound Tracker
# ---------------------------------------------------------------------------
class BranchAndBoundFragmentTracker:
    """Branch & Bound fragment linker for multi-object tracking.

    Parameters
    ----------
    w_iou : float
        Weight for IoU dissimilarity term (1 - IoU).
    w_app : float
        Weight for appearance dissimilarity term (1 - cosine_similarity).
    w_gap : float
        Weight for temporal gap normalized penalty.
    w_birth : float
        Cost paid to start a new track (fragment with no predecessor).
    max_stitch_gap_frames : int
        Maximum allowed frame gap between fragment end and next fragment start.
    gap_normalization : {"linear", "sqrt", "log"}
        Normalization mode for temporal gap length.
    best_first : bool
        (Reserved for potential priority ordering; currently not altering logic.)
    use_preprocessor : bool
        If True, run GreedyPreprocessor to form fragments; else each detection
        becomes a singleton fragment.
    """

    def __init__(
        self,
        w_iou: float = 0.2,
        w_app: float = 1.0,
        w_gap: float = 0.01,
        w_birth: float = 10.0,
        max_stitch_gap_frames: int = 5 * 30,  # e.g. 5 seconds @ 30fps
        gap_normalization: str = "linear",   # or 'sqrt' / 'log'
        best_first: bool = True,
        use_preprocessor: bool = True,
    ) -> None:
        self.w_iou = w_iou
        self.w_app = w_app
        self.w_gap = w_gap
        self.w_birth = w_birth
        self.max_stitch_gap_frames = max_stitch_gap_frames
        self.gap_normalization = gap_normalization
        self.best_first = best_first
        self.use_preprocessor = use_preprocessor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def track_detections(
        self,
        detections: List[Detection],
        video_properties: Optional[VideoInfo] = None,
        return_debug: bool = False,
    ) -> List[Track] | Tuple[List[Track], Dict[str, Any]]:
        """Link detection fragments into full tracks via branch & bound.

        Parameters
        ----------
        detections : list[Detection]
            All detections (of potentially multiple objects) for the sequence.
        video_properties : VideoInfo | None
            Optional video metadata (fps, etc.) passed to preprocessor.
        return_debug : bool
            If True, also return a debug dictionary (succ/pred arrays, cost, etc.).

        Returns
        -------
        tracks : list[Track]
            Reconstructed tracks with concatenated detections.
        (tracks, debug) : tuple
            If return_debug is True.
        """
        if not detections:
            return [] if not return_debug else ([], {
                "succ": [], "pred": [], "best_cost": 0.0,
                "fragment_to_track": [], "fragment_order_in_track": []
            })

        # Fragmentation step
        if self.use_preprocessor:
            fragments = GreedyPreprocessor().track_detections(detections, video_properties)
        else:
            # Treat every detection as singleton fragment
            fragments = [
                Track(track_id=i, sorted_detections=[d])
                for i, d in enumerate(sorted(detections, key=lambda d: d.frame_idx))
            ]

        # Create metadata for each fragment
        metas: List[_FragMeta] = [
            _FragMeta(
                idx=i,
                start_frame=f.sorted_detections[0].frame_idx,
                end_frame=f.sorted_detections[-1].frame_idx,
                start_det=f.sorted_detections[0],
                end_det=f.sorted_detections[-1],
            )
            for i, f in enumerate(fragments)
        ]
        # Sort fragments by start frame for acyclicity & deterministic ordering
        metas.sort(key=lambda m: m.start_frame)
        fragments = [fragments[m.idx] for m in metas]  # reorder fragments
        for new_i, m in enumerate(metas):
            m.idx = new_i  # reindex after sort

        n = len(metas)

        # Precompute feasible predecessor edges for each fragment j
        predecessors: List[List[Tuple[int, float]]] = [[] for _ in metas]
        for j, mj in enumerate(metas):
            for i, mi in enumerate(metas[:j]):
                # Enforce strict temporal ordering
                if mi.end_frame >= mj.start_frame:
                    continue  # overlap or touching; require gap >= 0 (strict after)
                gap = mj.start_frame - mi.end_frame - 1
                if gap > self.max_stitch_gap_frames:
                    continue
                edge_cost = self._edge_cost(mi, mj, gap)
                predecessors[j].append((i, edge_cost))
            predecessors[j].sort(key=lambda x: x[1])  # ascending cost

        # Best incoming cost per fragment (used in lower bound)
        best_incoming = [min((c for _, c in preds), default=self.w_birth) for preds in predecessors]

        # Successor / predecessor arrays for building solution
        succ: List[Optional[int]] = [None] * n
        pred: List[Optional[int]] = [None] * n
        assigned = [False] * n  # fragment considered in recursion path

        best_solution: Dict[str, Any] = {
            'cost': math.inf,
            'succ': None,
            'pred': None,
        }

        # Heuristic expansion order: fragments with strongest preference to link
        order = list(range(n))
        order.sort(key=lambda j: (best_incoming[j] - self.w_birth, metas[j].start_frame))

        # ----------------------- Lower Bound Function -----------------------
        def lower_bound() -> float:
            lb = 0.0
            for j in range(n):
                if not assigned[j] and pred[j] is None:
                    # Fragment not yet decided; must incur at least birth or edge cost
                    lb += min(best_incoming[j], self.w_birth)
            return lb

        # --------------------- Full Cost Re-computation ---------------------
        def finalize_cost() -> float:
            total = 0.0
            # Birth costs
            for j in range(n):
                if pred[j] is None:
                    total += self.w_birth
            # Edge costs (each counted once when j has successor)
            for j in range(n):
                s = succ[j]
                if s is not None:
                    gap = metas[s].start_frame - metas[j].end_frame - 1
                    total += self._edge_cost(metas[j], metas[s], gap)
            return total

        # ----------------------- Recursive Branching ------------------------
        def branch(pos: int, current_birth_cost: float, current_edge_cost: float) -> None:
            current_cost = current_birth_cost + current_edge_cost
            lb = lower_bound()
            if current_cost + lb >= best_solution['cost']:
                return  # prune

            if pos == n:
                # All fragments assigned (some may be implicit track roots)
                final_cost = finalize_cost()
                if final_cost < best_solution['cost']:
                    best_solution['cost'] = final_cost
                    best_solution['succ'] = succ.copy()
                    best_solution['pred'] = pred.copy()
                return

            j = order[pos]
            if assigned[j]:
                branch(pos + 1, current_birth_cost, current_edge_cost)
                return

            # Option 1: Start a new track for fragment j
            assigned[j] = True
            branch(pos + 1, current_birth_cost + self.w_birth, current_edge_cost)
            assigned[j] = False

            # Option 2: Link j to a feasible predecessor p
            for p_idx, cost in predecessors[j]:
                # Predecessor must have no successor yet
                if succ[p_idx] is not None:
                    continue
                # Prevent cycles (temporal order already ensures)
                # If predecessor not yet assigned and also root -> we must pay its birth cost implicitly now
                implicit_birth = 0.0
                if not assigned[p_idx] and pred[p_idx] is None:
                    implicit_birth = self.w_birth

                # Apply link
                old_pred_j = pred[j]
                old_succ_p = succ[p_idx]
                prev_assigned_p = assigned[p_idx]

                assigned[j] = True
                pred[j] = p_idx
                succ[p_idx] = j
                assigned[p_idx] = True  # ensure predecessor considered

                branch(pos + 1, current_birth_cost + implicit_birth, current_edge_cost + cost)

                # Undo link (backtrack)
                pred[j] = old_pred_j
                succ[p_idx] = old_succ_p
                assigned[j] = False
                if not prev_assigned_p:
                    assigned[p_idx] = False

        # Launch recursion
        branch(pos=0, current_birth_cost=0.0, current_edge_cost=0.0)

        # ----------------------- Reconstruction Phase -----------------------
        if best_solution['succ'] is None:
            logging.warning("Branch&Bound failed to find solution; returning identity mapping.")
            identity_tracks = [
                Track(track_id=i, sorted_detections=fragments[i].sorted_detections)
                for i in range(n)
            ]
            return identity_tracks if not return_debug else (
                identity_tracks,
                {
                    "succ": [None]*n,
                    "pred": [None]*n,
                    "best_cost": float('inf'),
                    "fragment_to_track": list(range(n)),
                    "fragment_order_in_track": [0]*n,
                    "fragment_start_frames": [m.start_frame for m in metas],
                    "fragment_end_frames": [m.end_frame for m in metas],
                },
            )

        final_tracks: List[Track] = []
        visited = [False] * n
        track_id = 0
        fragment_to_track = [-1]*n
        fragment_order_in_track = [-1]*n

        for i in range(n):
            if best_solution['pred'][i] is None and not visited[i]:
                chain = []
                cur = i
                order_in_track = 0
                while cur is not None and not visited[cur]:
                    chain.append(cur)
                    visited[cur] = True
                    fragment_to_track[cur] = track_id
                    fragment_order_in_track[cur] = order_in_track
                    order_in_track += 1
                    cur = best_solution['succ'][cur]
                # Concatenate detections from the fragments in this chain
                dets = []
                for idx in chain:
                    dets.extend(fragments[idx].sorted_detections)
                final_tracks.append(Track(track_id=track_id, sorted_detections=dets))
                track_id += 1

        if not return_debug:
            return final_tracks

        debug = {
            "succ": best_solution['succ'],
            "pred": best_solution['pred'],
            "best_cost": best_solution['cost'],
            "fragment_to_track": fragment_to_track,
            "fragment_order_in_track": fragment_order_in_track,
            "fragment_start_frames": [m.start_frame for m in metas],
            "fragment_end_frames": [m.end_frame for m in metas],
        }
        return final_tracks, debug

    # ------------------------------------------------------------------
    # Edge Cost Function
    # ------------------------------------------------------------------
    def _edge_cost(self, fa: _FragMeta, fb: _FragMeta, gap: int) -> float:
        """Compute cost of linking fragment A -> fragment B.

        Parameters
        ----------
        fa, fb : _FragMeta
            Metadata for predecessor and successor fragments.
        gap : int
            Number of *missing* frames between fa.end_frame and fb.start_frame
            (0 if consecutive frames, 1 means one skipped frame, etc.).

        Returns
        -------
        cost : float
            Weighted sum of normalized dissimilarities.
        """
        iou = fa.end_det.bbox.iou(fb.start_det.bbox)  # assumed in [0,1]
        cos = cosine_similarity(fa.end_det.feat, fb.start_det.feat)  # assumed in [-1,1]

        iou_term = (1.0 - max(0.0, min(1.0, iou))) * self.w_iou
        app_term = (1.0 - max(-1.0, min(1.0, cos))) * self.w_app

        if self.max_stitch_gap_frames <= 0:
            gap_norm = 0.0
        else:
            if self.gap_normalization == 'linear':
                gap_norm = gap / self.max_stitch_gap_frames
            elif self.gap_normalization == 'sqrt':
                gap_norm = math.sqrt(gap / self.max_stitch_gap_frames) if gap > 0 else 0.0
            elif self.gap_normalization == 'log':
                gap_norm = math.log1p(gap) / math.log1p(self.max_stitch_gap_frames) if gap > 0 else 0.0
            else:
                gap_norm = 0.0  # unknown mode -> treat as zero
        gap_term = gap_norm * self.w_gap

        return iou_term + app_term + gap_term


# ---------------------------------------------------------------------------
# Optional Utility: Human-readable Track Description
# ---------------------------------------------------------------------------
def describe_tracks(tracks: List[Track]) -> str:
    """Generate a human-readable multiline summary of tracks."""
    lines = []
    for t in tracks:
        if not t.sorted_detections:
            lines.append(f"Track {t.track_id}: <empty>")
            continue
        frames = [d.frame_idx for d in t.sorted_detections]
        lines.append(f"Track {t.track_id}: frames {frames[0]}–{frames[-1]} ({len(frames)} detections)")
    return "\n".join(lines)


__all__ = [
    'BranchAndBoundFragmentTracker',
    'describe_tracks',
]
