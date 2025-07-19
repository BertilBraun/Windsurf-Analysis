#!/usr/bin/env python3
"""
Branch-and-Bound (Backtracking) Fragment Linker for Multi-Object Tracking (MOT)
-----------------------------------------------------------------------------
Adds an *edge-level momentum change penalty* to discourage identity switches
that introduce implausible motion discontinuities.

Momentum Penalty Overview
=========================
We estimate per-fragment *representative velocities* (momentum, with unit mass):
  • first_vel  – velocity between the first two detections in the fragment
  • last_vel   – velocity between the last two detections in the fragment

At each candidate edge (fragment A -> fragment B) we add a normalized penalty:

Mode 'edge' (default): direct velocity discontinuity
    rel = ||v_b(start) - v_a(end)|| / (||v_a|| + ||v_b|| + eps)   ∈ [0,1]
    momentum_term = w_momentum * rel

Mode 'predict': compare predecessor's last velocity AND successor's initial
velocity to a *bridge* velocity required to translate A's last center to B's
first center across the temporal gap (gap+1 frames). We average the two
relative discrepancies (each normalized like above) to form the penalty.

Both modes keep the term within [0, w_momentum]. Set w_momentum=0.0 to disable.

Integration
===========
The penalty is added inside _edge_cost so it stays additive and compatible with
existing branch & bound bounding logic (we do NOT include it in the lower bound,
only realized edges pay it).

Note: For single-detection fragments, momentum cannot be estimated; such
fragments contribute 0 momentum penalty when used as source/target (graceful
fallback).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

from video_io import VideoInfo
from common_types import Detection, FrameIndex, Track, TrackId, cosine_similarity
from greedy_preprocessor import GreedyPreprocessor


@dataclass
class _FragMeta:
    idx: int
    start_frame: int
    end_frame: int
    start_det: Detection
    end_det: Detection
    first_vel: Optional[tuple[float, float]]  # (dx, dy) between first two detections
    last_vel: Optional[tuple[float, float]]   # (dx, dy) between last two detections


class BranchAndBoundFragmentTracker:
    def __init__(
        self,
        w_iou: float = 1.0,
        w_app: float = 1.0,
        w_gap: float = 0.01,
        w_birth: float = 10.0,
        w_momentum: float = 0.5,          # weight for momentum change penalty (0 disables)
        momentum_mode: str = "edge",       # 'edge' or 'predict'
        max_stitch_gap_frames: int = 5 * 30,  # 5 seconds @ 30fps
        gap_normalization: str = "linear",   # 'linear' | 'sqrt' | 'log'
        best_first: bool = True,
        use_preprocessor: bool = True,
        debug_costs: bool = True,          # if True, log per-edge term contributions and global breakdown
    ) -> None:
        self.w_iou = w_iou
        self.w_app = w_app
        self.w_gap = w_gap
        self.w_birth = w_birth
        self.w_momentum = w_momentum
        self.momentum_mode = momentum_mode
        self.max_stitch_gap_frames = max_stitch_gap_frames
        self.gap_normalization = gap_normalization
        self.best_first = best_first
        self.use_preprocessor = use_preprocessor
        self.debug_costs = debug_costs
        self._edge_cost_cache: dict[tuple[int,int,int], dict] = {}  # (fa, fb, gap) -> term breakdown

    # ------------------------------ Public API ------------------------------
    def track_detections(
        self,
        detections: List[Detection],
        video_properties: VideoInfo | None = None,
    ) -> List[Track]:
        logging.info("Branch & Bound Fragment Tracker: tracking %d detections", len(detections))
        if not detections:
            return []
        if self.use_preprocessor:
            fragments = GreedyPreprocessor().track_detections(detections, video_properties)
        else:
            # Treat every detection as a singleton fragment
            fragments = [
                Track(track_id=i, sorted_detections=[d])
                for i, d in enumerate(sorted(detections, key=lambda d: d.frame_idx))
            ]

        # Build fragment metadata including representative velocities
        metas: List[_FragMeta] = []
        for i, f in enumerate(fragments):
            dets = f.sorted_detections
            if len(dets) >= 2:
                # First velocity
                d0, d1 = dets[0], dets[1]
                x0 = 0.5 * (d0.bbox.x1 + d0.bbox.x2)
                y0 = 0.5 * (d0.bbox.y1 + d0.bbox.y2)
                x1p = 0.5 * (d1.bbox.x1 + d1.bbox.x2)
                y1p = 0.5 * (d1.bbox.y1 + d1.bbox.y2)
                first_vel = (x1p - x0, y1p - y0)
                # Last velocity
                d_lm1, d_l = dets[-2], dets[-1]
                xl1 = 0.5 * (d_lm1.bbox.x1 + d_lm1.bbox.x2)
                yl1 = 0.5 * (d_lm1.bbox.y1 + d_lm1.bbox.y2)
                xl = 0.5 * (d_l.bbox.x1 + d_l.bbox.x2)
                yl = 0.5 * (d_l.bbox.y1 + d_l.bbox.y2)
                last_vel = (xl - xl1, yl - yl1)
            else:
                first_vel = None
                last_vel = None
            metas.append(
                _FragMeta(
                    idx=i,
                    start_frame=dets[0].frame_idx,
                    end_frame=dets[-1].frame_idx,
                    start_det=dets[0],
                    end_det=dets[-1],
                    first_vel=first_vel,
                    last_vel=last_vel,
                )
            )

        # Sort by start time for temporal ordering
        metas.sort(key=lambda m: m.start_frame)
        fragments = [fragments[m.idx] for m in metas]
        for new_i, m in enumerate(metas):
            m.idx = new_i

        # Precompute feasible incoming edges (predecessors) for each fragment
        predecessors: List[List[tuple[int, float]]] = [[] for _ in metas]
        for j, mj in enumerate(metas):
            for i, mi in enumerate(metas[:j]):
                if mi.end_frame >= mj.start_frame:  # overlap or touch (need strictly after)
                    continue
                gap = mj.start_frame - mi.end_frame - 1
                if gap > self.max_stitch_gap_frames:
                    continue
                edge_cost = self._edge_cost(mi, mj, gap)
                predecessors[j].append((i, edge_cost))
            predecessors[j].sort(key=lambda x: x[1])

        best_incoming = [min([c for _, c in preds], default=self.w_birth) for preds in predecessors]

        # Branch & Bound State
        n = len(metas)
        succ = [None] * n  # successor index or None
        pred = [None] * n  # predecessor index or None
        assigned = [False] * n
        best_solution = {
            'cost': math.inf,
            'succ': None,
            'pred': None,
        }

        order = list(range(n))
        # Heuristic ordering: fragments with smallest (best_incoming - w_birth) first
        order.sort(key=lambda j: (best_incoming[j] - self.w_birth, metas[j].start_frame))

        def lower_bound():
            lb = 0.0
            for j in range(n):
                if not assigned[j]:
                    if pred[j] is None:
                        lb += min(best_incoming[j], self.w_birth)
            return lb

        def finalize_cost():
            total = 0.0
            for j in range(n):
                if pred[j] is None:
                    total += self.w_birth
                if succ[j] is not None:
                    gap = metas[succ[j]].start_frame - metas[j].end_frame - 1
                    total += self._edge_cost(metas[j], metas[succ[j]], gap, log_edge=False)
            return total

        def branch(pos: int, current_birth_cost: float, current_edge_cost: float):
            current_cost = current_birth_cost + current_edge_cost
            lb = lower_bound()
            if current_cost + lb >= best_solution['cost']:
                return
            if pos == n:
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

            # OPTION 1: Start new track at j
            assigned[j] = True
            branch(pos + 1, current_birth_cost + self.w_birth, current_edge_cost)
            assigned[j] = False

            # OPTION 2: Link j to a predecessor p
            for p_idx, cost in predecessors[j]:
                if assigned[p_idx] and succ[p_idx] is not None:
                    continue
                if succ[p_idx] is not None:
                    continue
                implicit_birth = 0.0
                if not assigned[p_idx] and pred[p_idx] is None:
                    implicit_birth = self.w_birth
                assigned[j] = True
                old_pred_j = pred[j]
                old_succ_p = succ[p_idx]
                pred[j] = p_idx
                succ[p_idx] = j
                prev_assigned_p = assigned[p_idx]
                assigned[p_idx] = True
                branch(
                    pos + 1,
                    current_birth_cost + implicit_birth,
                    current_edge_cost + cost,
                )
                pred[j] = old_pred_j
                succ[p_idx] = old_succ_p
                assigned[j] = False
                if not prev_assigned_p:
                    assigned[p_idx] = False

        branch(pos=0, current_birth_cost=0.0, current_edge_cost=0.0)

        # Debug global cost breakdown before reconstruction
        if self.debug_costs and best_solution['succ'] is not None:
            births = [i for i in range(n) if best_solution['pred'][i] is None]
            births_cost = len(births) * self.w_birth
            agg = {
                'birth': births_cost,
                'iou': 0.0,
                'app': 0.0,
                'gap': 0.0,
                'momentum': 0.0,
                'total': births_cost,
            }
            for a in range(n):
                b = best_solution['succ'][a]
                if b is not None:
                    gap = metas[b].start_frame - metas[a].end_frame - 1
                    terms = self._edge_cost(metas[a], metas[b], gap, return_terms=True, log_edge=False)
                    agg['iou'] += terms['iou_term']
                    agg['app'] += terms['app_term']
                    agg['gap'] += terms['gap_term']
                    agg['momentum'] += terms['momentum_term']
                    agg['total'] += terms['total']
            logging.debug("GLOBAL COST BREAKDOWN: %s", agg)
            for root in births:
                chain=[]
                cur=root
                chain_iou=chain_app=chain_gap=chain_mom=0.0
                while cur is not None:
                    nxt = best_solution['succ'][cur]
                    if nxt is not None:
                        gap = metas[nxt].start_frame - metas[cur].end_frame - 1
                        terms = self._edge_cost(metas[cur], metas[nxt], gap, return_terms=True, log_edge=False)
                        chain_iou += terms['iou_term']
                        chain_app += terms['app_term']
                        chain_gap += terms['gap_term']
                        chain_mom += terms['momentum_term']
                    chain.append(cur)
                    cur = nxt
                logging.debug(
                    "TRACK root_frag=%d chain=%s cost: birth=%.3f iou=%.3f app=%.3f gap=%.3f momentum=%.3f total=%.3f",
                    root, chain, self.w_birth, chain_iou, chain_app, chain_gap, chain_mom,
                    self.w_birth + chain_iou + chain_app + chain_gap + chain_mom,
                )

        # Reconstruct tracks
        if best_solution['succ'] is None:
            logging.warning("Branch&Bound failed; returning identity mapping.")
            return [
                Track(track_id=i, sorted_detections=fragments[i].sorted_detections)
                for i in range(len(fragments))
            ]

        final_tracks: List[Track] = []
        visited = [False] * n
        track_id = 0
        for i in range(n):
            if best_solution['pred'][i] is None and not visited[i]:
                chain = []
                cur = i
                while cur is not None and not visited[cur]:
                    chain.append(cur)
                    visited[cur] = True
                    cur = best_solution['succ'][cur]
                dets = []
                for idx in chain:
                    dets.extend(fragments[idx].sorted_detections)
                final_tracks.append(Track(track_id=track_id, sorted_detections=dets))
                track_id += 1
        return final_tracks

    # Edge cost between fragment A and fragment B (A end -> B start)
    def _edge_cost(self, fa: _FragMeta, fb: _FragMeta, gap: int, *, return_terms: bool = False, log_edge: bool = True):
        """Compute edge cost and optionally return per-term breakdown.

        Parameters
        ----------
        fa, fb : _FragMeta
            Predecessor and successor fragment metadata.
        gap : int
            Number of empty frames between fa.end and fb.start minus 1 (>=0).
        return_terms : bool
            If True, return a dict with each term; otherwise return total float cost.
        log_edge : bool
            If True and debug_costs enabled, emit a logging.debug line.
        """
        iou = fa.end_det.bbox.iou(fb.start_det.bbox)
        cos = cosine_similarity(fa.end_det.feat, fb.start_det.feat)
        # Individual terms --------------------------------------------------
        iou_clamped = max(0.0, min(1.0, iou))
        cos_clamped = max(-1.0, min(1.0, cos))
        iou_term = (1.0 - iou_clamped) * self.w_iou
        app_term = (1.0 - cos_clamped) * self.w_app

        if self.gap_normalization == 'linear':
            gap_norm = gap / self.max_stitch_gap_frames if self.max_stitch_gap_frames > 0 else 0.0
        elif self.gap_normalization == 'sqrt':
            gap_norm = math.sqrt(gap / self.max_stitch_gap_frames) if self.max_stitch_gap_frames > 0 else 0.0
        elif self.gap_normalization == 'log':
            gap_norm = math.log1p(gap) / math.log1p(self.max_stitch_gap_frames) if self.max_stitch_gap_frames > 0 else 0.0
        else:
            gap_norm = 0.0
        gap_term = gap_norm * self.w_gap

        momentum_term = 0.0
        if self.w_momentum > 0.0:
            eps = 1e-6
            va = fa.last_vel
            vb = fb.first_vel
            if self.momentum_mode == 'predict':
                dt = gap + 1 if gap >= 0 else 1
                xa = 0.5 * (fa.end_det.bbox.x1 + fa.end_det.bbox.x2)
                ya = 0.5 * (fa.end_det.bbox.y1 + fa.end_det.bbox.y2)
                xb = 0.5 * (fb.start_det.bbox.x1 + fb.start_det.bbox.x2)
                yb = 0.5 * (fb.start_det.bbox.y1 + fb.start_det.bbox.y2)
                vp = ((xb - xa) / dt, (yb - ya) / dt)
                rel_a = rel_b = 0.0
                if va is not None:
                    diff_a = math.hypot(vp[0] - va[0], vp[1] - va[1])
                    norm_a = math.hypot(vp[0], vp[1]) + math.hypot(va[0], va[1]) + eps
                    rel_a = diff_a / norm_a
                if vb is not None:
                    diff_b = math.hypot(vp[0] - vb[0], vp[1] - vb[1])
                    norm_b = math.hypot(vp[0], vp[1]) + math.hypot(vb[0], vb[1]) + eps
                    rel_b = diff_b / norm_b
                momentum_term = 0.5 * (rel_a + rel_b) * self.w_momentum
            else:  # 'edge'
                if va is not None and vb is not None:
                    diff = math.hypot(vb[0] - va[0], vb[1] - va[1])
                    denom = math.hypot(va[0], va[1]) + math.hypot(vb[0], vb[1]) + eps
                    rel = diff / denom
                    momentum_term = rel * self.w_momentum

        total = iou_term + app_term + gap_term + momentum_term
        terms = {
            'fa': fa.idx,
            'fb': fb.idx,
            'gap': gap,
            'iou_raw': iou,
            'cos_raw': cos,
            'iou_term': iou_term,
            'app_term': app_term,
            'gap_norm': gap_norm,
            'gap_term': gap_term,
            'momentum_term': momentum_term,
            'total': total,
        }
        # Cache breakdown
        self._edge_cost_cache[(fa.idx, fb.idx, gap)] = terms
        if self.debug_costs and log_edge:
            logging.debug(
                "EDGE fa=%d->fb=%d gap=%d iou=%.3f iou_term=%.3f cos=%.3f app_term=%.3f gap_norm=%.3f gap_term=%.3f momentum_term=%.3f total=%.3f",
                fa.idx, fb.idx, gap, iou, iou_term, cos, app_term, gap_norm, gap_term, momentum_term, total
            )
        if return_terms:
            return terms
        return total
