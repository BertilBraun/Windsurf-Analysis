from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ARTIFACT_AUG_ENABLED = True
ARTIFACT_DUPLICATE_PROB = 0.60
ARTIFACT_SECOND_DUPLICATE_PROB = 0.0
ARTIFACT_STRENGTH_WEIGHTS = {"mild": 0.50, "medium": 0.35, "strong": 0.15}


@dataclass(frozen=True)
class ArtifactAugConfig:
    enabled: bool = ARTIFACT_AUG_ENABLED
    duplicate_prob: float = ARTIFACT_DUPLICATE_PROB
    second_duplicate_prob: float = ARTIFACT_SECOND_DUPLICATE_PROB
    strength_weights: dict[str, float] | None = None


DEFAULT_CONFIG = ArtifactAugConfig(strength_weights=ARTIFACT_STRENGTH_WEIGHTS)


def write_image_with_artifact_variants(
    src_img_path: Path,
    dst_dir: Path,
    out_stem: str | None = None,
    rng: random.Random | None = None,
    is_train: bool = True,
    config: ArtifactAugConfig = DEFAULT_CONFIG,
) -> list[Path]:
    """Write the original image and optional degraded variants. Returns written image paths."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_stem or src_img_path.stem
    ext = src_img_path.suffix.lower()
    rng = rng or random.Random()

    original_out = dst_dir / f"{out_stem}{ext}"
    if src_img_path.resolve() != original_out.resolve():
        original_out.write_bytes(src_img_path.read_bytes())
    written_paths = [original_out]

    if not (is_train and config.enabled):
        return written_paths

    weights = (config.strength_weights or ARTIFACT_STRENGTH_WEIGHTS).copy()
    if not weights:
        return written_paths

    n_aug = 0
    if rng.random() < float(config.duplicate_prob):
        n_aug += 1
    if rng.random() < float(config.second_duplicate_prob):
        n_aug += 1
    if n_aug == 0:
        return written_paths

    img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
    if img is None:
        return written_paths

    severities = _sample_severities(rng, weights, n_aug)
    for idx, severity in enumerate(severities, start=1):
        degraded = degrade_image(img, rng, severity)
        aug_out = dst_dir / f"{out_stem}_aug_{idx}_{severity}{ext}"
        if not cv2.imwrite(str(aug_out), degraded):
            continue
        written_paths.append(aug_out)
    return written_paths


def degrade_image(img: np.ndarray, rng: random.Random, severity: str) -> np.ndarray:
    out = img.copy()
    params = _severity_params(severity)

    out = _downscale_upscale(out, rng, params["scale_min"], params["scale_max"])

    if rng.random() < params["blur_prob"]:
        out = _gaussian_blur(out, rng, params["blur_sigma_min"], params["blur_sigma_max"])

    if rng.random() < params["chroma_prob"]:
        out = _chroma_soften(out, rng, params["chroma_scale_min"], params["chroma_scale_max"])

    if rng.random() < params["compress_prob"]:
        out = _recompress(out, rng, params["quality_min"], params["quality_max"])

    if rng.random() < params["noise_prob"]:
        out = _add_noise(out, rng, params["noise_std_min"], params["noise_std_max"])

    return out


def _sample_severities(rng: random.Random, weights: dict[str, float], n: int) -> list[str]:
    names = [k for k, v in weights.items() if v > 0]
    vals = [float(weights[k]) for k in names]
    if not names:
        return []
    return rng.choices(names, weights=vals, k=n)


def _severity_params(severity: str) -> dict[str, float]:
    if severity == "mild":
        return {
            "scale_min": 0.82,
            "scale_max": 0.97,
            "quality_min": 45,
            "quality_max": 70,
            "blur_prob": 0.35,
            "blur_sigma_min": 0.2,
            "blur_sigma_max": 0.8,
            "chroma_prob": 0.35,
            "chroma_scale_min": 0.60,
            "chroma_scale_max": 0.90,
            "compress_prob": 1.0,
            "noise_prob": 0.10,
            "noise_std_min": 1.0,
            "noise_std_max": 3.0,
        }
    if severity == "strong":
        return {
            "scale_min": 0.45,
            "scale_max": 0.75,
            "quality_min": 12,
            "quality_max": 35,
            "blur_prob": 0.75,
            "blur_sigma_min": 0.8,
            "blur_sigma_max": 2.0,
            "chroma_prob": 0.80,
            "chroma_scale_min": 0.25,
            "chroma_scale_max": 0.55,
            "compress_prob": 1.0,
            "noise_prob": 0.20,
            "noise_std_min": 2.0,
            "noise_std_max": 5.0,
        }
    # medium/default
    return {
        "scale_min": 0.62,
        "scale_max": 0.86,
        "quality_min": 25,
        "quality_max": 50,
        "blur_prob": 0.55,
        "blur_sigma_min": 0.4,
        "blur_sigma_max": 1.4,
        "chroma_prob": 0.60,
        "chroma_scale_min": 0.35,
        "chroma_scale_max": 0.70,
        "compress_prob": 1.0,
        "noise_prob": 0.15,
        "noise_std_min": 1.5,
        "noise_std_max": 4.0,
    }


def _downscale_upscale(img: np.ndarray, rng: random.Random, scale_min: float, scale_max: float) -> np.ndarray:
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return img
    scale = float(rng.uniform(scale_min, scale_max))
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))
    down_interp = rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
    up_interp = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    small = cv2.resize(img, (small_w, small_h), interpolation=down_interp)
    return cv2.resize(small, (w, h), interpolation=up_interp)


def _gaussian_blur(img: np.ndarray, rng: random.Random, sigma_min: float, sigma_max: float) -> np.ndarray:
    sigma = float(rng.uniform(sigma_min, sigma_max))
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _chroma_soften(img: np.ndarray, rng: random.Random, scale_min: float, scale_max: float) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    h, w = img.shape[:2]
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    scale = float(rng.uniform(scale_min, scale_max))
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    cr_small = cv2.resize(cr, (sw, sh), interpolation=cv2.INTER_AREA)
    cb_small = cv2.resize(cb, (sw, sh), interpolation=cv2.INTER_AREA)
    cr_up = cv2.resize(cr_small, (w, h), interpolation=cv2.INTER_LINEAR)
    cb_up = cv2.resize(cb_small, (w, h), interpolation=cv2.INTER_LINEAR)

    ycrcb_out = np.dstack([y, cr_up, cb_up]).astype(np.uint8, copy=False)
    return cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)


def _recompress(img: np.ndarray, rng: random.Random, quality_min: int, quality_max: int) -> np.ndarray:
    quality = int(round(rng.uniform(quality_min, quality_max)))
    if rng.random() < 0.8:
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), max(5, min(95, quality))]
    else:
        ext = ".webp"
        params = [int(cv2.IMWRITE_WEBP_QUALITY), max(5, min(95, quality))]
    ok, enc = cv2.imencode(ext, img, params)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def _add_noise(img: np.ndarray, rng: random.Random, std_min: float, std_max: float) -> np.ndarray:
    std = float(rng.uniform(std_min, std_max))
    if std <= 0:
        return img
    noise = np.random.default_rng(rng.randrange(0, 2**32)).normal(0.0, std, img.shape)
    noisy = img.astype(np.float32) + noise.astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)
