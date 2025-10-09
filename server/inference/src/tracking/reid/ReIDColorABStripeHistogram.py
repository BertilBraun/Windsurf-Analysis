from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from server.inference.src.util.algebra import l1_normalize, l2_normalize
from server.inference.src.util.similarity_helpers import HellingerEmbedding


# -----------------------------
# Configuration dataclasses
# -----------------------------


@dataclass(frozen=True)
class MaskParams:
    """Parameters for background-vs-foreground masking from border statistics."""

    side_strip_fraction: float = 0.2  # width of left/right strips used for BG model & stats
    use_side_strips_only: bool = True  # build BG model only from left/right strips (ignore top/bottom)
    gaussian_blur_ksize: int = 11  # pre-blur kernel (odd or <=1 to disable)
    ab_clip: float = 64.0  # clamp Lab a/b channel to [-ab_clip, +ab_clip]
    tau_quantile: float = 0.80  # Mahalanobis^2 threshold quantile on border pixels
    chi2_min: float = 3.0
    chi2_max: float = 10.0
    grad_quantile: float = 0.5  # gradient magnitude quantile on outer ring for smoothness gate
    dilate_px: int = 1  # slight dilation to recover thin structures


@dataclass(frozen=True)
class HistogramParams:
    """Parameters for histograms."""

    bins_ab: int = 16  # (a,b) joint histogram bins
    bins_L: int = 16  # L channel bins
    bins_h: int = 36  # hue bins (circular), 10° per bin
    ab_sigma: float = 1.2  # Gaussian smoothing in ab joint histogram


@dataclass(frozen=True)
class StripeParams:
    """Parameters for vertical stripe partitioning."""

    n_stripes: int = 3


@dataclass(frozen=True)
class WeightParams:
    """Weights & pixel weighting for descriptor construction."""

    include_L: bool = True
    sat_gamma: float = 1.2  # saturation^gamma pixel weighting
    w_ab: float = 1.0
    w_h: float = 0.7
    w_L: float = 0.2
    eps: float = 1e-8


# -----------------------------
# Image transforms & utilities
# -----------------------------


def _maybe_blur_bgr(bgr: np.ndarray, ksize: int) -> np.ndarray:
    if ksize is None or ksize <= 1:
        return bgr
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(bgr, (ksize, ksize), 0)


def _bgr_to_lab_hsv(bgr: np.ndarray, ab_clip: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns L(0..100), a[-clip,clip], b[-clip,clip], hue_u8(0..179)"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = (lab[..., 0] / 255.0) * 100.0
    a = np.clip(lab[..., 1] - 128.0, -ab_clip, ab_clip)
    b = np.clip(lab[..., 2] - 128.0, -ab_clip, ab_clip)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_u8 = hsv[..., 0].astype(np.uint8)
    return L, a, b, hue_u8


def _saturation_weights(bgr: np.ndarray, gamma: float) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    return np.power(S, float(gamma), dtype=np.float32)


def _side_strip_mask(shape: Tuple[int, int], ring_fraction: float) -> np.ndarray:
    h, w = shape
    tw = max(1, int(round(w * ring_fraction)))
    m = np.zeros((h, w), dtype=bool)
    m[:, :tw] = True
    m[:, -tw:] = True
    return m


def _vertical_stripes(shape: Tuple[int, int], n: int) -> List[np.ndarray]:
    h, w = shape
    stripes = []
    for i in range(n):
        y0 = (i * h) // n
        y1 = ((i + 1) * h) // n
        band = np.zeros((h, w), dtype=bool)
        band[y0:y1, :] = True
        stripes.append(band)
    return stripes


# -----------------------------
# Background model & mask
# -----------------------------


def _robust_bg_ab_model(a: np.ndarray, b: np.ndarray, model_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Median -> 80% inlier mean/cov for ab; returns (mu[2], cov[2x2])."""
    ab = np.stack([a[model_mask], b[model_mask]], axis=1)
    if ab.size == 0:
        # degenerate fallback
        return np.array([0.0, 0.0], np.float32), (np.eye(2, dtype=np.float32) * 10.0)
    mu_med = np.median(ab, axis=0)
    d2e = ((ab - mu_med) ** 2).sum(axis=1)
    keep = d2e <= np.quantile(d2e, 0.80)
    ab_in = ab[keep] if keep.any() else ab
    mu = ab_in.mean(axis=0).astype(np.float32)
    cov = np.cov(ab_in.T).astype(np.float32)
    cov += np.eye(2, dtype=np.float32) * 1e-3
    return mu, cov


def _mahalanobis_d2(a: np.ndarray, b: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(cov).astype(np.float32)
    x0 = (a - mu[0]).astype(np.float32)
    x1 = (b - mu[1]).astype(np.float32)
    return inv[0, 0] * x0 * x0 + 2.0 * inv[0, 1] * x0 * x1 + inv[1, 1] * x1 * x1


def _gradient_magnitude_u8(gray_f32: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag


def _components_touching_border(mask_bool: np.ndarray) -> np.ndarray:
    num, lab = cv2.connectedComponents(mask_bool.astype(np.uint8))
    if num <= 1:
        return mask_bool
    h, w = mask_bool.shape
    frame = np.zeros((h, w), dtype=bool)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
    touching = np.unique(lab[frame & (lab > 0)])
    return np.isin(lab, touching)


def build_fg_mask_from_border(
    bgr: np.ndarray,
    mask_params: MaskParams,
) -> np.ndarray:
    """
    Build a foreground mask using a border-derived ab color model and a smoothness gate.
    """
    # pre-blur to stabilize gradients/color
    bgr_blur = _maybe_blur_bgr(bgr, mask_params.gaussian_blur_ksize)

    # color spaces
    L, a, b, _ = _bgr_to_lab_hsv(bgr_blur, mask_params.ab_clip)

    # masks for model and candidate region
    model_mask = (
        _side_strip_mask(bgr.shape[:2], mask_params.side_strip_fraction)
        if mask_params.use_side_strips_only
        else np.ones(bgr.shape[:2], dtype=bool)
    )

    # robust ab background model
    mu, cov = _robust_bg_ab_model(a, b, model_mask)
    d2 = _mahalanobis_d2(a, b, mu, cov)

    # smoothness (low texture) gate from gradient magnitude
    gray = cv2.cvtColor(bgr_blur, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_mag = _gradient_magnitude_u8(gray)
    g_tau = float(np.quantile(grad_mag[model_mask], mask_params.grad_quantile))

    # Mahalanobis threshold from model region; clamp to chi2 bounds
    tau = float(np.quantile(d2[model_mask], mask_params.tau_quantile))
    tau = max(mask_params.chi2_min, min(mask_params.chi2_max, tau))

    # candidates likely to be background (color like border AND smooth)
    bg_candidates = (d2 <= tau) & (grad_mag <= g_tau)

    # retain only components connected to image border → BG
    bg_mask = _components_touching_border(bg_candidates)

    # foreground = complement
    fg_mask = ~bg_mask

    # morphological cleanup
    if mask_params.dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mask_params.dilate_px + 1, 2 * mask_params.dilate_px + 1))
        fg_mask = cv2.dilate(fg_mask.astype(np.uint8), k, iterations=1) > 0

    # Anything not connected to the largest component is background
    num, lab = cv2.connectedComponents(fg_mask.astype(np.uint8))
    if num > 1:
        largest_component = lab == np.argmax(np.bincount(lab.flatten())[1:]) + 1
        fg_mask = largest_component

    # fallback: if mask is tiny, keep all
    if fg_mask.mean() < 0.05:
        fg_mask[:] = True
    return fg_mask


# -----------------------------
# Histograms
# -----------------------------


def hist2d_ab(
    a: np.ndarray,
    b: np.ndarray,
    sample_mask: np.ndarray,
    bins_ab: int,
    ab_clip: float,
    weights: Optional[np.ndarray],
    sigma: float,
) -> np.ndarray:
    if sample_mask.sum() == 0:
        return np.zeros((bins_ab * bins_ab,), np.float32)
    H, _, _ = np.histogram2d(
        a[sample_mask].ravel(),
        b[sample_mask].ravel(),
        bins=bins_ab,
        range=[[-ab_clip, ab_clip], [-ab_clip, ab_clip]],
        weights=None if weights is None else weights[sample_mask].ravel(),
    )
    H = H.astype(np.float32)
    if sigma > 0:
        H = cv2.GaussianBlur(H, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)
    return H.ravel()


def hist1d_L(
    L: np.ndarray,
    sample_mask: np.ndarray,
    bins_L: int,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    if sample_mask.sum() == 0:
        return np.zeros((bins_L,), np.float32)
    H, _ = np.histogram(
        L[sample_mask].ravel(),
        bins=bins_L,
        range=(0.0, 100.0),
        weights=None if weights is None else weights[sample_mask].ravel(),
    )
    return H.astype(np.float32)


def circ_hist1d_hue(
    hue_u8: np.ndarray,
    sample_mask: np.ndarray,
    bins_h: int,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    if sample_mask.sum() == 0:
        return np.zeros((bins_h,), np.float32)
    h_scaled = hue_u8[sample_mask].astype(np.float32) / 180.0 * bins_h  # [0, bins)
    i_floor = np.floor(h_scaled).astype(np.int32)
    frac = h_scaled - i_floor
    i0 = (i_floor) % bins_h
    i1 = (i_floor + 1) % bins_h
    w = np.ones_like(frac, np.float32) if weights is None else weights[sample_mask].astype(np.float32)
    hist = np.zeros((bins_h,), np.float32)
    np.add.at(hist, i0, (1.0 - frac) * w)
    np.add.at(hist, i1, frac * w)
    return hist


# -----------------------------
# Descriptor builder
# -----------------------------


class ReIDColorABStripeHistogram:
    """
    Joint descriptor = concat over stripes and a global block of:
      - sqrt(L1-normalized ab 2D histogram) * w_ab
      - sqrt(L1-normalized hue circular histogram) * w_h
      - optional sqrt(L1-normalized L histogram) * w_L
    Then L2-normalize the full vector.
    """

    def __init__(
        self,
        masks: MaskParams = MaskParams(),
        hists: HistogramParams = HistogramParams(),
        stripes: StripeParams = StripeParams(),
        weights: WeightParams = WeightParams(),
        use_mask: bool = True,
        pre_blur_crop_ksize: int = 3,
    ):
        self.masks = masks
        self.hists = hists
        self.stripes = stripes
        self.weights = weights
        self.use_mask = bool(use_mask)
        self.pre_blur_crop_ksize = int(pre_blur_crop_ksize)

    # -------------------------

    def _compute_mask(self, bgr: np.ndarray) -> np.ndarray:
        return build_fg_mask_from_border(bgr, self.masks) if self.use_mask else np.ones(bgr.shape[:2], bool)

    def _build_block(
        self,
        L: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        hue_u8: np.ndarray,
        sample_mask: np.ndarray,
        pix_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """One block = ab + hue (+ optional L), sqrt(L1-normed), weighted by w_*."""
        eps = self.weights.eps
        ab = (
            hist2d_ab(a, b, sample_mask, self.hists.bins_ab, self.masks.ab_clip, pix_weights, self.hists.ab_sigma) + eps
        )
        hue = circ_hist1d_hue(hue_u8, sample_mask, self.hists.bins_h, pix_weights) + eps
        parts = [np.sqrt(l1_normalize(ab)) * self.weights.w_ab, np.sqrt(l1_normalize(hue)) * self.weights.w_h]
        if self.weights.include_L:
            Lh = hist1d_L(L, sample_mask, self.hists.bins_L, pix_weights) + eps
            parts.append(np.sqrt(l1_normalize(Lh)) * self.weights.w_L)
        return np.concatenate(parts, axis=0)

    def _embedding_for_crop(self, crop_bgr: np.ndarray) -> HellingerEmbedding:
        assert crop_bgr.ndim == 3 and crop_bgr.shape[2] == 3 and crop_bgr.size > 0

        # light blur to stabilize color/edges
        bgr = _maybe_blur_bgr(crop_bgr, self.pre_blur_crop_ksize)

        # mask & per-pixel weights
        fg_mask = self._compute_mask(bgr)
        pix_w = _saturation_weights(bgr, self.weights.sat_gamma)

        # color spaces
        L, a, b, hue_u8 = _bgr_to_lab_hsv(bgr, self.masks.ab_clip)

        # stripe blocks
        blocks = []
        for band in _vertical_stripes(fg_mask.shape, self.stripes.n_stripes):
            m = fg_mask & band
            if m.sum() < 20:
                m = fg_mask  # fallback if band is too small
            blocks.append(self._build_block(L, a, b, hue_u8, m, pix_w))

        # global block
        blocks.append(self._build_block(L, a, b, hue_u8, fg_mask, pix_w))

        vec = np.concatenate(blocks, axis=0)
        if not np.isfinite(vec).all() or vec.sum() == 0:
            vec = np.ones_like(vec, np.float32)

        # NOTE: Visualization is disabled for now
        # self._visualize(crop_bgr, fg_mask, L, a, b)

        return HellingerEmbedding(l2_normalize(vec))

    # public API
    def get_features_for_crops(self, crops: List[np.ndarray]) -> Sequence[HellingerEmbedding]:
        return [self._embedding_for_crop(crop) for crop in crops]

    def _visualize(self, crop_bgr: np.ndarray, fg_mask: np.ndarray, L: np.ndarray, a: np.ndarray, b: np.ndarray):
        import matplotlib.pyplot as plt

        h, w = crop_bgr.shape[:2]

        crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        overlay = crop_bgr.copy()
        green = np.zeros_like(overlay)
        green[..., 1] = fg_mask
        ov = cv2.addWeighted(overlay, 0.7, green, 0.3, 0)
        for i in range(self.stripes.n_stripes):
            y = (i + 1) * h // self.stripes.n_stripes
            cv2.line(ov, (0, y), (w, y), (255, 0, 0), 1)

        masked = crop_bgr.copy()
        masked[fg_mask == 0] = 0

        cols = 5
        rows = 1
        ax = plt.subplot(rows, cols, 1)
        ax.imshow(crop_bgr)
        ax.set_title('Original')
        ax.axis('off')
        ax = plt.subplot(rows, cols, 2)
        ax.imshow(ov)
        ax.set_title(f'Mask overlay | stripes={self.stripes.n_stripes}')
        ax.axis('off')
        ax = plt.subplot(rows, cols, 3)
        ax.imshow(masked)
        ax.set_title('Masked crop')
        ax.axis('off')
        ax = plt.subplot(rows, cols, 4)
        abH = hist2d_ab(a, b, fg_mask, self.hists.bins_ab, self.masks.ab_clip, None, self.hists.ab_sigma)
        abN = abH / (abH.max() + 1e-12)
        abN = abN.reshape(self.hists.bins_ab, self.hists.bins_ab)
        ax.imshow(abN, origin='lower', interpolation='nearest', aspect='equal')
        ax.set_title('ab joint hist (global)')
        ax.axis('off')
        ax = plt.subplot(rows, cols, 5)
        Lh = hist1d_L(L, fg_mask, self.hists.bins_L, None)
        nn = Lh / (Lh.max() + 1e-12)
        ax.bar(np.arange(len(nn)), nn, width=0.9)
        ax.set_title('L hist (global)')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()
