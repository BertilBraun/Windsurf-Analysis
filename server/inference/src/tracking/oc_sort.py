from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment

import numpy as np

from ..common_types import BoundingBox, Detection, Point, Track
from ..tracking.tracking import Tracker
from ..util.video_io import VideoInfo
from ..visualization.stabilize import Transform

# =========================
# Math helpers
# =========================


def rotmat(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def apply_cmc_center(pt: Point, dx: float, dy: float, angle_deg: float, img_cx: float, img_cy: float) -> Point:
    v = np.array([pt.x - img_cx, pt.y - img_cy], dtype=np.float64)
    v = rotmat(math.radians(angle_deg)).dot(v)
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


# =========================
# Kalman box filter (SORT-style)
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
    vdir: Optional[np.ndarray] = None  # unit 2D
    gallery: List[np.ndarray] = field(default_factory=list)  # embeddings
    last_det: Optional[Detection] = None  # full detection object of last match


class OCSortEmbed:
    """OC-SORT with observation-centric virtual boxes, BYTE second pass, OCR re-association, embeddings fusion, and CMC."""

    def __init__(
        self,
        det_hi: float = 0.6,
        det_lo: float = 0.1,
        iou_thr: float = 0.3,
        inertia: float = 0.2,
        delta_t: int = 3,
        max_age: int = 30,
        min_hits: int = 3,
        use_byte: bool = True,
        gallery_size: int = 10,
        reid_sim_thr: float = 0.8,
        iou_gate_min: float = 0.05,
        ambig_iou_margin: float = 0.1,
    ):
        self.det_hi, self.det_lo = det_hi, det_lo
        self.iou_thr = iou_thr
        self.inertia = inertia
        self.delta_t = delta_t
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_byte = use_byte
        self.gallery_size = gallery_size
        self.reid_sim_thr = reid_sim_thr
        self.iou_gate_min = iou_gate_min
        self.ambig_iou_margin = ambig_iou_margin

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
        T: Transform,
    ) -> List[Tuple[BoundingBox, int]]:
        self.t = frame_idx
        self.img_cx, self.img_cy = (img_w - 1) / 2.0, (img_h - 1) / 2.0

        # 1) CMC then predict
        for tr in self.tracks:
            c = tr.kf.bbox_from_x(tr.kf.x).center
            c_cmc = apply_cmc_center(c, T.dx, T.dy, T.da, self.img_cx, self.img_cy)
            tr.kf.x[0] += c_cmc.x - c.x
            tr.kf.x[1] += c_cmc.y - c.y
            if tr.vdir is not None:
                vdir = tr.vdir
                vdir = rotmat(math.radians(T.da)).dot(vdir)
                vdir /= np.linalg.norm(vdir) + 1e-8
                tr.vdir = vdir
            tr.kf.predict()

        # 2) split detections
        D_hi = [d for d in detections if d.confidence >= self.det_hi]
        D_lo = [d for d in detections if self.det_lo <= d.confidence < self.det_hi]

        # 3) Stage-A: OC virtual boxes + fused cost
        M_A, U_hi, U_trk = self._assoc_oc_fused(D_hi)

        self._apply_matches(D_hi, M_A)
        # unmatched tracks carry over to next stages
        # 4) BYTE-style pass on low-score detections, fused to break ties
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

        # 7) Init new tracks from remaining high-score detections
        for i in U_hi:
            self._start_track(D_hi[i])

        # 8) Ageing
        alive: List[Tracklet] = []
        for tr in self.tracks:
            if tr.last_seen == self.t:
                tr.miss = 0
                tr.hits += 1
                if tr.hits >= self.min_hits:
                    tr.confirmed = True
            else:
                tr.miss += 1
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

    def _assoc_oc_fused(self, dets: List[Detection]):
        if not dets or not self.tracks:
            return [], list(range(len(dets))), list(range(len(self.tracks)))

        virt: List[BoundingBox] = []
        for tr in self.tracks:
            pred = tr.kf.bbox_from_x(tr.kf.x)
            last = tr.last_obs
            vdir = self._estimate_vdir(tr)
            step = last.center.distance_to(pred.center)
            step = min(
                step, 2.0 * last.diagonal_length
            )  # clamp step to 2x the diagonal length to not explode after long misses
            shift = self.inertia * step * vdir
            vb = bbox_from_center_wh(last.center.x + shift[0], last.center.y + shift[1], last.width, last.height)
            virt.append(vb)

        I = np.zeros((len(dets), len(self.tracks)), dtype=np.float64)
        S = np.zeros_like(I)
        L = np.zeros_like(I)
        for i, d in enumerate(dets):
            for j, tr in enumerate(self.tracks):
                I[i, j] = d.bbox.iou(virt[j])
                sim = max(cosine(d.embedding, g) for g in tr.gallery) if tr.gallery else 0.0
                S[i, j] = sim

        # adaptive lambda: shift weight to embeddings when IoU is ambiguous or track has gaps
        iou_max = I.max(axis=1, keepdims=True)
        iou_2nd = np.partition(I, -2, axis=1)[:, -2][:, None] if I.shape[1] >= 2 else np.zeros_like(iou_max)
        ambig = (iou_max - iou_2nd) < self.ambig_iou_margin
        for i in range(len(dets)):
            for j, tr in enumerate(self.tracks):
                lam = 0.6
                if tr.miss >= 5:
                    lam = 0.35
                if ambig[i, 0] or I[i, j] < 0.2:
                    lam = min(lam, 0.35)
                L[i, j] = lam

        big = 1e6
        C = L * (1.0 - I) + (1.0 - L) * (1.0 - S)
        for i, d in enumerate(dets):
            for j, tr in enumerate(self.tracks):
                # weak spatial gate
                if (
                    I[i, j] < self.iou_gate_min
                    and d.bbox.center.distance_to(tr.kf.bbox_from_x(tr.kf.x).center) > 3.0 * tr.last_obs.diagonal_length
                ):
                    C[i, j] = big

        ri, ci = linear_sum_assignment(C)
        matches: List[Tuple[int, int]] = []
        U_det: List[int] = []
        U_trk: List[int] = []
        matched_det, matched_trk = set(), set()
        for i in range(len(dets)):
            if i not in ri:
                U_det.append(i)
        for j in range(len(self.tracks)):
            if j not in ci:
                U_trk.append(j)
        for r, c in zip(ri, ci):
            iou_ok = I[r, c] >= self.iou_thr
            sim_ok = S[r, c] >= self.reid_sim_thr if not iou_ok else True
            if iou_ok or sim_ok:
                matches.append((r, c))
                matched_det.add(r)
                matched_trk.add(c)
            else:
                if r not in matched_det:
                    U_det.append(r)
                if c not in matched_trk:
                    U_trk.append(c)
        return matches, U_det, U_trk

    def _assoc_byte_fused(self, dets_lo: List[Detection], u_trk_idx: List[int]):
        if not dets_lo or not u_trk_idx:
            return [], list(range(len(dets_lo))), u_trk_idx[:]
        I = np.zeros((len(dets_lo), len(u_trk_idx)))
        S = np.zeros_like(I)
        L = np.zeros_like(I)
        for i, d in enumerate(dets_lo):
            for jj, idx in enumerate(u_trk_idx):
                tr = self.tracks[idx]
                I[i, jj] = d.bbox.iou(tr.kf.bbox_from_x(tr.kf.x))
                S[i, jj] = max(cosine(d.embedding, g) for g in tr.gallery) if tr.gallery else 0.0
                lam = 0.5 if tr.miss >= 3 else 0.6
                if I[i, jj] < 0.2:
                    lam = min(lam, 0.3)
                L[i, jj] = lam
        C = L * (1.0 - I) + (1.0 - L) * (1.0 - S)
        ri, ci = linear_sum_assignment(C)
        matches: List[Tuple[int, int]] = []
        U_det: List[int] = []
        U_trk: List[int] = []
        matched_det, matched_trk = set(), set()
        for i in range(len(dets_lo)):
            if i not in ri:
                U_det.append(i)
        for j in range(len(u_trk_idx)):
            if j not in ci:
                U_trk.append(u_trk_idx[j])
        for r, c in zip(ri, ci):
            # accept if IoU ok OR strong appearance
            if (I[r, c] >= self.iou_thr - 0.05) or (S[r, c] >= self.reid_sim_thr):
                matches.append((r, u_trk_idx[c]))
                matched_det.add(r)
                matched_trk.add(c)
            else:
                if r not in matched_det:
                    U_det.append(r)
                if c not in matched_trk:
                    U_trk.append(u_trk_idx[c])
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
            if I[r, c] >= self.iou_thr - 0.1:
                matches.append((U_hi[r], U_trk[c]))
                keep_hi.discard(r)
                keep_trk.discard(c)
        U_hi2 = [U_hi[i] for i in sorted(keep_hi)]
        U_trk2 = [U_trk[i] for i in sorted(keep_trk)]
        return matches, U_hi2, U_trk2

    def _assoc_ocr(self, dets: List[Detection], u_trk_idx: List[int]):
        if not dets or not u_trk_idx:
            return [], list(range(len(dets))), u_trk_idx[:]
        I = np.zeros((len(dets), len(u_trk_idx)))
        for i, d in enumerate(dets):
            for j, ti in enumerate(u_trk_idx):
                I[i, j] = d.bbox.iou(self.tracks[ti].last_obs)
        C = 1.0 - I
        ri, ci = linear_sum_assignment(C)
        matches: List[Tuple[int, int]] = []
        U_det: List[int] = []
        keep = set(range(len(u_trk_idx)))
        for r, c in zip(ri, ci):
            if I[r, c] >= self.iou_thr - 0.1:
                matches.append((r, u_trk_idx[c]))
                keep.discard(c)
            else:
                U_det.append(r)
        U_trk = [u_trk_idx[j] for j in sorted(keep)]
        # unmatched dets
        for i in range(len(dets)):
            if i not in ri and i not in U_det:
                U_det.append(i)
        return matches, U_det, U_trk

    def _reid_revive(
        self, D_hi: List[Detection], U_hi: List[int], D_lo: List[Detection], U_lo: List[int], U_trk: List[int]
    ):
        # build full detection list index space
        dets_all = [D_hi[i] for i in U_hi] + [D_lo[i] for i in U_lo]
        if not dets_all or not U_trk:
            return [], [], []
        # similarity matrix
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
            if S[r, c] >= self.reid_sim_thr:
                # map back to global det index
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
            # vdir from k-previous observation
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
            # gallery
            tr.gallery.append(d.embedding.astype(np.float64))
            if len(tr.gallery) > self.gallery_size:
                tr.gallery.pop(0)

    def _start_track(self, det: Detection) -> None:
        cx, cy, s, r = KalmanBox.z_from_bbox(det.bbox)
        kf = KalmanBox(cx, cy, s, r)
        tr = Tracklet(
            tid=self.next_tid,
            kf=kf,
            last_obs=det.bbox.copy(),
            last_seen=self.t,
            obs_hist={self.t: det.bbox.copy()},
            hits=1,
            miss=0,
            confirmed=False,
            vdir=None,
            gallery=[det.embedding.astype(np.float64)],
            last_det=det,
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
        det_lo: float = 0.1,
        iou_thr: float = 0.3,
        inertia: float = 0.2,
        delta_t: int = 3,
        max_age_seconds: float = 10.0,
        min_hits: int = 2,
        use_byte: bool = True,
        reid_sim_thr: float = 0.8,
        gallery_size: int = 10,
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
        )
        self.max_age_seconds = max_age_seconds

    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        # group single-detection "tracks" into per-frame lists
        by_frame: Dict[int, List[Detection]] = {}
        for t in tracks:
            assert len(t.sorted_detections) == 1, 'Input must be one detection per Track.'
            d = t.sorted_detections[0]
            by_frame.setdefault(d.frame_idx, []).append(d)

        # map frame -> transform (identity default)
        T_by_frame: Dict[int, Transform] = {tr.frame_idx: tr for tr in transforms}

        def get_T(f: int) -> Transform:
            assert f in T_by_frame, f'Transform not found for frame {f}, {list(T_by_frame.keys())}'
            return T_by_frame[f]

        W, H, FPS = video_properties.width, video_properties.height, video_properties.fps
        oc_max_age = int(round(self.max_age_seconds * max(1, FPS)))

        core = OCSortEmbed(max_age=oc_max_age, **self.params)  # type: ignore

        tid2dets: Dict[int, List[Detection]] = {}

        for f in range(video_properties.total_frames):
            dets_f = by_frame.get(f, [])
            T = get_T(f)
            core.update(dets_f, f, W, H, T)

            # collect matches/inits for this frame
            for tr in core.tracks:
                if tr.last_seen == f and tr.last_det is not None:
                    tid2dets.setdefault(tr.tid, []).append(tr.last_det)

        # build final tracks
        out_tracks: List[Track] = []
        for tid, dets in tid2dets.items():
            dets.sort(key=lambda d: d.frame_idx)
            out_tracks.append(Track(track_id=tid, sorted_detections=dets))
        return out_tracks
