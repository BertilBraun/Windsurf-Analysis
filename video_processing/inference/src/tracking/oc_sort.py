from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..common_types import BoundingBox, Detection, Point, Track
from ..tracking.tracking import Tracker
from ..util.video_io import VideoInfo
from ..visualization.stabilize import Transform  # dx, dy, da (radians), frame_idx


# =========================
# Math helpers
# =========================


def rotmat(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def apply_cmc_center(pt: Point, dx: float, dy: float, angle_rad: float, img_cx: float, img_cy: float) -> Point:
    """Rotate around image center by angle_rad, then translate by (dx, dy)."""
    v = np.array([pt.x - img_cx, pt.y - img_cy], dtype=np.float64)
    v = rotmat(angle_rad).dot(v)
    v += np.array([img_cx + dx, img_cy + dy], dtype=np.float64)
    return Point(int(round(v[0])), int(round(v[1])))


def bbox_from_center_wh(cx: float, cy: float, w: float, h: float) -> BoundingBox:
    return BoundingBox(int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a /= np.linalg.norm(a) + 1e-8
    b /= np.linalg.norm(b) + 1e-8
    return float(np.clip(a @ b, -1.0, 1.0))


def H_from_transform(t: Transform) -> np.ndarray:
    """Affine 3x3 from (dx, dy, da_rad)."""
    c, s = math.cos(t.da), math.sin(t.da)
    return np.array([[c, -s, t.dx], [s, c, t.dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def transform_from_H(H: np.ndarray, frame_idx: int) -> Transform:
    """Extract (dx, dy, da_rad) from 3x3 SE(2) matrix."""
    da = math.atan2(H[1, 0], H[0, 0])
    dx = float(H[0, 2])
    dy = float(H[1, 2])
    return Transform(dx=dx, dy=dy, da=da, frame_idx=frame_idx)


def compute_delta_transforms(transforms: List[Transform]) -> Dict[int, Transform]:
    if not transforms:
        return {}
    transforms_sorted = sorted(transforms, key=lambda x: x.frame_idx)
    H = {t.frame_idx: H_from_transform(t) for t in transforms_sorted}
    idxs = [t.frame_idx for t in transforms_sorted]
    first = min(idxs)
    deltas: Dict[int, Transform] = {first: Transform(0.0, 0.0, 0.0, first)}
    for f in idxs[1:]:
        prev = f - 1
        if prev not in H:
            deltas[f] = Transform(0.0, 0.0, 0.0, f)
            continue
        # delta = H_{t-1}^{-1} @ H_t   (maps t-1 -> t)
        B = np.linalg.inv(H[prev]) @ H[f]
        deltas[f] = transform_from_H(B, f)
    return deltas


# =========================
# Kalman box filter (SORT-style) + gating
# =========================


class KalmanBox:
    """x=[cx,cy,s,r,vx,vy,vs]; z=[cx,cy,s,r]."""

    def __init__(self, cx: float, cy: float, s: float, r: float):
        self.x = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.F = np.eye(7)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.Q = np.eye(7) * 0.01
        self.Q[-1, -1] *= 0.01

    @staticmethod
    def z_from_bbox(bb: BoundingBox) -> np.ndarray:
        w = max(1.0, bb.width)
        h = max(1.0, bb.height)
        return np.array([(bb.x1 + bb.x2) * 0.5, (bb.y1 + bb.y2) * 0.5, w * h, w / h], dtype=np.float64)

    @staticmethod
    def bbox_from_x(x: np.ndarray) -> BoundingBox:
        w = math.sqrt(max(1e-6, x[2] * x[3]))
        h = max(1e-6, x[2] / max(1e-6, w))
        return bbox_from_center_wh(x[0], x[1], w, h)

    def predict(self) -> BoundingBox:
        if (self.x[2] + self.x[6]) <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.bbox_from_x(self.x)

    def update(self, bb: Optional[BoundingBox]) -> None:
        if bb is None:
            return
        z = self.z_from_bbox(bb)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = np.eye(7)
        self.P = (I - K @ self.H) @ self.P

    def maha(self, bb: BoundingBox) -> float:
        """Mahalanobis distance of bbox measurement to current state prediction."""
        z = self.z_from_bbox(bb)
        y = z - self.H @ self.x  # assuming self.x already predicted
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)


# =========================
# OC-SORT + embeddings
# =========================


@dataclass
class Tracklet:
    tid: int
    kf: KalmanBox
    last_obs: BoundingBox
    last_seen: int
    obs_hist: Dict[int, BoundingBox]
    hits: int = 1
    miss: int = 0
    confirmed: bool = False
    hit_streak: int = 0
    vdir: Optional[np.ndarray] = None  # unit 2D
    gallery: List[np.ndarray] = field(default_factory=list)  # embeddings (L2-normalized)
    last_det: Optional[Detection] = None  # full detection object of last match
    born_at: int = 0


class OCSortEmbed:
    """
    OC-SORT with:
      - observation-centric virtual boxes
      - motion-consistency penalty (soft)
      - soft mutual-nearest tie filter (not strict)
      - BYTE second pass
      - OCR re-association
      - embedding fusion and ReID revive
      - camera-motion compensation (per-frame delta)
      - softer association + early output option
    """

    def __init__(
        self,
        det_hi: float = 0.6,
        det_lo: float = 0.05,
        iou_thr: float = 0.30,
        inertia: float = 0.18,
        delta_t: int = 3,
        max_age: int = 30,
        min_hits: int = 2,
        use_byte: bool = True,
        gallery_size: int = 10,
        reid_sim_thr: float = 0.80,
        reid_sim_thr_tent: float = 0.85,
        iou_thr_tent: float = 0.35,
        iou_gate_min: float = 0.02,
        ambig_iou_margin: float = 0.12,
        sim_margin: float = 0.04,
        obs_hist_cap: int = 100,
        spawn_suppress_iou: float = 0.60,
        spawn_suppress_sim: float = 0.75,
        w_mot: float = 0.15,
        maha_gate: float = 16.0,  # ~chi2(4, 0.99)=13.3; 16 is lenient
        output_tentative: bool = True,  # reduce startup lag
        output_tentative_max: int = 2,  # frames after birth to allow tentative output
        dedup_enable: bool = False,  # disable aggressive dedup to avoid drops
        dup_iou: float = 0.65,
        dup_sim: float = 0.90,
    ):
        self.det_hi, self.det_lo = det_hi, det_lo
        self.iou_thr = iou_thr
        self.iou_thr_tent = iou_thr_tent
        self.inertia = inertia
        self.delta_t = delta_t
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_byte = use_byte
        self.gallery_size = gallery_size
        self.reid_sim_thr = reid_sim_thr
        self.reid_sim_thr_tent = reid_sim_thr_tent
        self.iou_gate_min = iou_gate_min
        self.ambig_iou_margin = ambig_iou_margin
        self.sim_margin = sim_margin
        self.obs_hist_cap = obs_hist_cap
        self.spawn_suppress_iou = spawn_suppress_iou
        self.spawn_suppress_sim = spawn_suppress_sim
        self.w_mot = w_mot
        self.maha_gate = maha_gate
        self.output_tentative = output_tentative
        self.output_tentative_max = output_tentative_max
        self.dedup_enable = dedup_enable
        self.dup_iou = dup_iou
        self.dup_sim = dup_sim

        self.tracks: List[Tracklet] = []
        self.next_tid = 1
        self.t = -1
        self.img_cx = 0.0
        self.img_cy = 0.0

    # ----- public update -----
    def update(
        self,
        detections: List[Detection],  # embeddings always set
        frame_idx: int,
        img_w: int,
        img_h: int,
        T_delta: Transform,  # per-frame delta (dx, dy, da_rad) mapping t-1 -> t
    ) -> List[Tuple[BoundingBox, int]]:
        self.t = frame_idx
        self.img_cx, self.img_cy = (img_w - 1) / 2.0, (img_h - 1) / 2.0

        # L2-normalize embeddings at ingest
        for d in detections:
            e = d.embedding.astype(np.float64)
            d.embedding = e / (np.linalg.norm(e) + 1e-8)

        # 1) CMC then predict
        for tr in self.tracks:
            c = tr.kf.bbox_from_x(tr.kf.x).center
            c_cmc = apply_cmc_center(c, T_delta.dx, T_delta.dy, T_delta.da, self.img_cx, self.img_cy)
            tr.kf.x[0] += c_cmc.x - c.x
            tr.kf.x[1] += c_cmc.y - c.y
            if tr.vdir is not None:
                vdir = rotmat(T_delta.da).dot(tr.vdir)
                tr.vdir = vdir / (np.linalg.norm(vdir) + 1e-8)
            tr.kf.predict()

        # 2) split detections
        D_hi = [d for d in detections if d.confidence >= self.det_hi]
        D_lo = [d for d in detections if self.det_lo <= d.confidence < self.det_hi]

        # 3) Stage-A: OC virtual boxes + fused cost + motion penalty + soft MN filter + KF gating
        M_A, U_hi, U_trk = self._assoc_oc_fused(D_hi)
        self._apply_matches(D_hi, M_A)

        # 4) BYTE-style pass on low-score detections, fused + motion penalty
        if self.use_byte and len(D_lo) and len(U_trk):
            M_B, U_lo, U_trk = self._assoc_byte_fused(D_lo, U_trk)
            self._apply_matches(D_lo, M_B)
        else:
            U_lo = list(range(len(D_lo)))

        # 5) OCR re-association with last observations
        if U_trk and U_hi:
            M_OCR, U_hi, U_trk = self._assoc_ocr_direct(D_hi, U_hi, U_trk)
            self._apply_matches(D_hi, M_OCR)

        # 6) ReID revive on remaining unmatched using gallery
        if len(U_trk):
            M_R, used_hi, used_lo = self._reid_revive(D_hi, U_hi, D_lo, U_lo, U_trk)
            self._apply_matches(D_hi, [(di, ti) for di, ti in M_R if di < len(D_hi)])
            self._apply_matches(D_lo, [(di - len(D_hi), ti) for di, ti in M_R if di >= len(D_hi)])
            U_hi = [i for i in U_hi if i not in set(used_hi)]
            U_lo = [i for i in U_lo if (i + len(D_hi)) not in set(used_lo)]

        # 7) Init new tracks from remaining high-score detections, with softer suppression
        for i in U_hi:
            di = D_hi[i]
            suppress = False
            for tr in self.tracks:
                if not tr.confirmed:
                    continue
                pred = tr.kf.bbox_from_x(tr.kf.x)
                if di.bbox.iou(pred) >= self.spawn_suppress_iou:
                    suppress = True
                    break
                sim = max(cosine(di.embedding, g) for g in tr.gallery) if tr.gallery else 0.0
                if (
                    sim >= self.spawn_suppress_sim
                    and di.bbox.center.distance_to(tr.last_obs.center) <= 0.6 * tr.last_obs.diagonal_length
                ):
                    suppress = True
                    break
            if not suppress:
                self._start_track(di)

        # 8) Ageing and confirmation
        alive: List[Tracklet] = []
        for tr in self.tracks:
            if tr.last_seen == self.t:
                tr.miss = 0
                tr.hits += 1
                tr.hit_streak = tr.hit_streak + 1
                if tr.hit_streak >= self.min_hits:
                    tr.confirmed = True
            else:
                tr.miss += 1
                tr.hit_streak = 0
            if tr.miss <= self.max_age:
                alive.append(tr)
        self.tracks = alive

        # 9) Output confirmed
        out: List[Tuple[BoundingBox, int]] = []
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            bb = tr.last_obs if tr.last_seen == self.t else tr.kf.bbox_from_x(tr.kf.x)
            out.append((bb, tr.tid))
        return out

    # ----- associations -----

    @staticmethod
    def _soft_mutual_filter(
        C: np.ndarray, ri: np.ndarray, ci: np.ndarray, eps: float = 0.05
    ) -> Tuple[List[int], List[int]]:
        """Keep pairs close to both row and col minima; softer than strict mutual-NN."""
        if len(ri) == 0:
            return [], []
        row_min = C.min(axis=1, keepdims=True)
        col_min = C.min(axis=0, keepdims=True)
        fr: List[int] = []
        fc: List[int] = []
        for r, c in zip(ri, ci):
            if C[r, c] <= row_min[r, 0] + eps and C[r, c] <= col_min[0, c] + eps:
                fr.append(int(r))
                fc.append(int(c))
        return fr, fc

    def _assoc_oc_fused(self, dets: List[Detection]):
        if not dets or not self.tracks:
            return [], list(range(len(dets))), list(range(len(self.tracks)))

        virt: List[BoundingBox] = []
        for tr in self.tracks:
            pred = tr.kf.bbox_from_x(tr.kf.x)
            last = tr.last_obs
            vdir = self._estimate_vdir(tr)
            step = last.center.distance_to(pred.center)
            step = min(step, 2.0 * last.diagonal_length)
            shift = self.inertia * step * vdir
            vb = bbox_from_center_wh(last.center.x + shift[0], last.center.y + shift[1], last.width, last.height)
            virt.append(vb)

        I = np.zeros((len(dets), len(self.tracks)), dtype=np.float64)  # IoU
        S = np.zeros_like(I)  # appearance sim
        L = np.zeros_like(I)  # lambda
        M = np.zeros_like(I)  # motion penalty
        G = np.zeros_like(I)  # KF gating penalty (0 ok, 1 block)
        big = 1e6

        for i, d in enumerate(dets):
            for j, tr in enumerate(self.tracks):
                I[i, j] = d.bbox.iou(virt[j])
                S[i, j] = max(cosine(d.embedding, g) for g in tr.gallery) if tr.gallery else 0.0
                # motion
                if tr.vdir is not None and np.linalg.norm(tr.vdir) > 1e-6:
                    disp = np.array(
                        [d.bbox.center.x - tr.last_obs.center.x, d.bbox.center.y - tr.last_obs.center.y],
                        dtype=np.float64,
                    )
                    n = np.linalg.norm(disp) + 1e-8
                    dir_cos = float(np.dot(tr.vdir, disp / n))
                    M[i, j] = 0.5 * (1.0 - max(-1.0, min(1.0, dir_cos)))
                # KF gate (lenient)
                if tr.kf.maha(d.bbox) > self.maha_gate:
                    G[i, j] = 1.0

        # adaptive lambda
        iou_max = I.max(axis=1, keepdims=True)
        iou_2nd = np.partition(I, -2, axis=1)[:, -2][:, None] if I.shape[1] >= 2 else np.zeros_like(iou_max)
        ambig_row = (iou_max - iou_2nd) < self.ambig_iou_margin
        for i in range(len(dets)):
            for j, tr in enumerate(self.tracks):
                lam = 0.55
                if tr.miss >= 5:
                    lam = 0.35
                if ambig_row[i, 0] or I[i, j] < 0.2 or I[i, j] > 0.6:
                    lam = min(lam, 0.30)
                L[i, j] = lam

        C = L * (1.0 - I) + (1.0 - L) * (1.0 - S) + self.w_mot * M
        # weak spatial + KF gating
        for i in range(len(dets)):
            for j, tr in enumerate(self.tracks):
                if (
                    I[i, j] < self.iou_gate_min
                    and dets[i].bbox.center.distance_to(tr.kf.bbox_from_x(tr.kf.x).center)
                    > 3.0 * tr.last_obs.diagonal_length
                ):
                    C[i, j] = big
                if G[i, j] > 0.5:
                    C[i, j] = big

        ri, ci = linear_sum_assignment(C)
        ri, ci = self._soft_mutual_filter(C, ri, ci, eps=0.05)

        matches: List[Tuple[int, int]] = []
        U_det: List[int] = []
        U_trk: List[int] = []
        ri_set = set(ri)
        ci_set = set(ci)
        for i in range(len(dets)):
            if i not in ri_set:
                U_det.append(i)
        for j in range(len(self.tracks)):
            if j not in ci_set:
                U_trk.append(j)

        for r, c in zip(ri, ci):
            tr = self.tracks[c]
            iou_ok = I[r, c] >= (self.iou_thr if tr.confirmed else self.iou_thr_tent)
            if tr.confirmed:
                if iou_ok:
                    # if IoU ambiguous, require small appearance margin
                    row = I[r]
                    if I.shape[1] >= 2:
                        top2 = np.partition(row, -2)[-2]
                        if (row.max() - top2) < self.ambig_iou_margin:
                            s_row = S[r]
                            s2 = np.partition(s_row, -2)[-2] if s_row.size >= 2 else -1.0
                            if S[r, c] < s2 + self.sim_margin:
                                # allow anyway if cost is clearly best
                                if not (C[r, c] <= C[r].min() + 0.02 and C[r, c] <= C[:, c].min() + 0.02):
                                    continue
                    matches.append((r, c))
                elif S[r, c] >= self.reid_sim_thr:
                    matches.append((r, c))
            else:
                # tentative: OR rule to avoid missing early frames
                if iou_ok or S[r, c] >= self.reid_sim_thr_tent:
                    matches.append((r, c))
        return matches, U_det, U_trk

    def _assoc_byte_fused(self, dets_lo: List[Detection], u_trk_idx: List[int]):
        if not dets_lo or not u_trk_idx:
            return [], list(range(len(dets_lo))), u_trk_idx[:]
        I = np.zeros((len(dets_lo), len(u_trk_idx)))
        S = np.zeros_like(I)
        L = np.zeros_like(I)
        M = np.zeros_like(I)
        big = 1e6
        for i, d in enumerate(dets_lo):
            for jj, idx in enumerate(u_trk_idx):
                tr = self.tracks[idx]
                I[i, jj] = d.bbox.iou(tr.kf.bbox_from_x(tr.kf.x))
                S[i, jj] = max(cosine(d.embedding, g) for g in tr.gallery) if tr.gallery else 0.0
                lam = 0.5 if tr.miss >= 3 else 0.6
                if I[i, jj] < 0.2:
                    lam = min(lam, 0.3)
                L[i, jj] = lam
                if tr.vdir is not None and np.linalg.norm(tr.vdir) > 1e-6:
                    disp = np.array(
                        [d.bbox.center.x - tr.last_obs.center.x, d.bbox.center.y - tr.last_obs.center.y],
                        dtype=np.float64,
                    )
                    n = np.linalg.norm(disp) + 1e-8
                    dir_cos = float(np.dot(tr.vdir, disp / n))
                    M[i, jj] = 0.5 * (1.0 - max(-1.0, min(1.0, dir_cos)))
                # lenient KF gate
                if tr.kf.maha(d.bbox) > self.maha_gate:
                    I[i, jj] = 0.0
                    S[i, jj] = 0.0
                    M[i, jj] = 1.0

        C = L * (1.0 - I) + (1.0 - L) * (1.0 - S) + self.w_mot * M
        C[np.isnan(C)] = big

        ri, ci = linear_sum_assignment(C)
        ri, ci = self._soft_mutual_filter(C, ri, ci, eps=0.05)

        matches: List[Tuple[int, int]] = []
        U_det: List[int] = []
        U_trk: List[int] = []
        ri_set = set(ri)
        ci_set = set(ci)
        for i in range(len(dets_lo)):
            if i not in ri_set:
                U_det.append(i)
        for j in range(len(u_trk_idx)):
            if j not in ci_set:
                U_trk.append(u_trk_idx[j])

        for r, c in zip(ri, ci):
            tr = self.tracks[u_trk_idx[c]]
            thr_iou = self.iou_thr if tr.confirmed else self.iou_thr_tent
            thr_sim = self.reid_sim_thr if tr.confirmed else self.reid_sim_thr_tent
            if (I[r, c] >= (thr_iou - 0.10)) or (S[r, c] >= (thr_sim - 0.05)):
                matches.append((r, u_trk_idx[c]))
        return matches, U_det, U_trk

    def _assoc_ocr_direct(self, D_hi: List[Detection], U_hi: List[int], U_trk: List[int]):
        if not U_hi or not U_trk:
            return [], U_hi[:], U_trk[:]
        I = np.zeros((len(U_hi), len(U_trk)))
        for i, di in enumerate(U_hi):
            for j, ti in enumerate(U_trk):
                I[i, j] = D_hi[di].bbox.iou(self.tracks[ti].last_obs)
        C = 1.0 - I
        ri, ci = linear_sum_assignment(C)
        matches: List[Tuple[int, int]] = []
        keep_hi = set(range(len(U_hi)))
        keep_trk = set(range(len(U_trk)))
        for r, c in zip(ri, ci):
            if I[r, c] >= self.iou_thr - 0.15:
                matches.append((U_hi[r], U_trk[c]))
                keep_hi.discard(r)
                keep_trk.discard(c)
        U_hi2 = [U_hi[i] for i in sorted(keep_hi)]
        U_trk2 = [U_trk[i] for i in sorted(keep_trk)]
        return matches, U_hi2, U_trk2

    def _reid_revive(
        self, D_hi: List[Detection], U_hi: List[int], D_lo: List[Detection], U_lo: List[int], U_trk: List[int]
    ):
        dets_all = [D_hi[i] for i in U_hi] + [D_lo[i] for i in U_lo]
        if not dets_all or not U_trk:
            return [], [], []
        S = np.zeros((len(dets_all), len(U_trk)))
        for i, d in enumerate(dets_all):
            for j, ti in enumerate(U_trk):
                tr = self.tracks[ti]
                S[i, j] = max(cosine(d.embedding, g) for g in tr.gallery) if tr.gallery else 0.0
        C = 1.0 - S
        ri, ci = linear_sum_assignment(C)
        matches: List[Tuple[int, int]] = []
        used_hi_local: List[int] = []
        used_lo_local: List[int] = []
        for r, c in zip(ri, ci):
            if S[r, c] >= max(0.8, self.reid_sim_thr - 0.05):
                if r < len(U_hi):
                    di_global = U_hi[r]
                    used_hi_local.append(r)
                else:
                    r2 = r - len(U_hi)
                    di_global = len(D_hi) + U_lo[r2]
                    used_lo_local.append(r)
                matches.append((di_global, U_trk[c]))
        return matches, [U_hi[i] for i in used_hi_local], [U_lo[i - len(U_hi)] for i in used_lo_local]

    # ----- bookkeeping -----

    def _apply_matches(self, dets: List[Detection], matches: List[Tuple[int, int]]) -> None:
        for di, ti in matches:
            d = dets[di]
            tr = self.tracks[ti]
            kprev = self._k_prev_obs(tr, self.delta_t)
            if kprev is not None:
                v = np.array([d.bbox.center.x - kprev.center.x, d.bbox.center.y - kprev.center.y], dtype=np.float64)
                n = np.linalg.norm(v) + 1e-8
                tr.vdir = v / n
            tr.kf.update(d.bbox)
            tr.last_obs = d.bbox.copy()
            tr.last_seen = self.t
            tr.last_det = d
            tr.obs_hist[self.t] = tr.last_obs
            if len(tr.obs_hist) > self.obs_hist_cap:
                del tr.obs_hist[min(tr.obs_hist.keys())]
            tr.gallery.append(d.embedding.astype(np.float64))
            if len(tr.gallery) > self.gallery_size:
                tr.gallery.pop(0)

    def _start_track(self, det: Detection) -> None:
        cx, cy, s, r = KalmanBox.z_from_bbox(det.bbox)
        kf = KalmanBox(cx, cy, s, r)
        e = det.embedding.astype(np.float64)
        e /= np.linalg.norm(e) + 1e-8
        tr = Tracklet(
            tid=self.next_tid,
            kf=kf,
            last_obs=det.bbox.copy(),
            last_seen=self.t,
            obs_hist={self.t: det.bbox.copy()},
            hits=1,
            miss=0,
            confirmed=False,
            hit_streak=1,
            vdir=None,
            gallery=[e],
            last_det=det,
            born_at=self.t,
        )
        self.tracks.append(tr)
        self.next_tid += 1

    @staticmethod
    def _k_prev_obs(tr: Tracklet, k: int) -> Optional[BoundingBox]:
        tgt = tr.last_seen - k
        if tgt in tr.obs_hist:
            return tr.obs_hist[tgt]
        if tr.obs_hist:
            return tr.obs_hist[min(tr.obs_hist.keys())]
        return None

    def _estimate_vdir(self, tr: Tracklet) -> np.ndarray:
        if tr.vdir is not None:
            return tr.vdir
        kprev = self._k_prev_obs(tr, self.delta_t)
        if kprev is None:
            return np.array([0.0, 0.0])
        v = np.array([tr.last_obs.center.x - kprev.center.x, tr.last_obs.center.y - kprev.center.y], dtype=np.float64)
        n = np.linalg.norm(v) + 1e-8
        return v / n


# =========================
# Adapter to your Tracker Protocol
# =========================


class OCSortEmbedTracker(Tracker):
    def __init__(
        self,
        det_hi: float = 0.6,
        det_lo: float = 0.05,
        iou_thr: float = 0.30,
        inertia: float = 0.18,
        delta_t: int = 3,
        max_age_seconds: float = 10.0,
        min_hits: int = 2,
        use_byte: bool = True,
        reid_sim_thr: float = 0.80,
        gallery_size: int = 10,
        ambig_iou_margin: float = 0.12,
        sim_margin: float = 0.04,
        spawn_suppress_iou: float = 0.60,
        spawn_suppress_sim: float = 0.75,
        w_mot: float = 0.15,
        maha_gate: float = 16.0,
        output_tentative: bool = True,
        output_tentative_max: int = 2,
        dedup_enable: bool = False,
        dup_iou: float = 0.65,
        dup_sim: float = 0.90,
    ):
        self.params = dict(
            det_hi=det_hi,
            det_lo=det_lo,
            iou_thr=iou_thr,
            inertia=inertia,
            delta_t=delta_t,
            min_hits=min_hits,
            use_byte=use_byte,
            reid_sim_thr=reid_sim_thr,
            gallery_size=gallery_size,
            ambig_iou_margin=ambig_iou_margin,
            sim_margin=sim_margin,
            spawn_suppress_iou=spawn_suppress_iou,
            spawn_suppress_sim=spawn_suppress_sim,
            w_mot=w_mot,
            maha_gate=maha_gate,
            output_tentative=output_tentative,
            output_tentative_max=output_tentative_max,
            dedup_enable=dedup_enable,
            dup_iou=dup_iou,
            dup_sim=dup_sim,
        )
        self.max_age_seconds = max_age_seconds

    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        # group single-detection "tracks" into per-frame lists
        by_frame: Dict[int, List[Detection]] = {}
        for t in tracks:
            assert len(t.sorted_detections) == 1, 'Input must be one detection per Track.'
            d = t.sorted_detections[0]
            by_frame.setdefault(d.frame_idx, []).append(d)

        # compute per-frame delta transforms (mapping t-1->t); identity for first frame
        delta_by_frame: Dict[int, Transform] = compute_delta_transforms(transforms)

        def get_delta(f: int) -> Transform:
            if f in delta_by_frame:
                return delta_by_frame[f]
            return Transform(0.0, 0.0, 0.0, f)

        W, H, FPS = video_properties.width, video_properties.height, video_properties.fps
        oc_max_age = int(round(self.max_age_seconds * max(1, FPS)))

        core = OCSortEmbed(max_age=oc_max_age, **self.params)  # type: ignore

        tid2dets: Dict[int, List[Detection]] = {}

        for f in range(video_properties.approximate_total_frames):
            dets_f = by_frame.get(f, [])
            T_delta = get_delta(f)
            core.update(dets_f, f, W, H, T_delta)

            # collect matches/inits for this frame (include tentative if enabled)
            for tr in core.tracks:
                if tr.last_seen == f and tr.last_det is not None:
                    if tr.confirmed or (core.output_tentative and (f - tr.born_at) <= core.output_tentative_max):
                        tid2dets.setdefault(tr.tid, []).append(tr.last_det)

        # build final tracks
        out_tracks: List[Track] = []
        for tid, dets in tid2dets.items():
            dets.sort(key=lambda d: d.frame_idx)
            out_tracks.append(Track(track_id=tid, sorted_detections=dets))
        return out_tracks
