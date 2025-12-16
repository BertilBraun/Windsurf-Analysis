from __future__ import annotations

import cv2
import numpy as np

from typing import Iterable, Optional, Tuple

from inference.src.common_types import BoundingBox


Image = np.ndarray


def new_canvas(shape: Tuple[int, int]) -> Image:
    return np.zeros(shape, dtype=np.uint8)


def draw_bounding_box(
    image: Image, bbox: BoundingBox, color: Tuple[int, int, int], thickness: int = 2, label: Optional[str] = None
) -> None:
    cv2.rectangle(image, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), color, int(thickness))
    if label:
        draw_text(image, label, (bbox.x1, max(0, bbox.y1 - 6)))


def draw_text(
    image: Image,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        (int(position[0]), int(position[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(font_scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def draw_arrow(
    image: Image, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2
) -> None:
    cv2.arrowedLine(
        image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, int(thickness), tipLength=0.3
    )


def compose_side_by_side(*sides, target_height: int | None = None) -> Image:
    if target_height is not None and target_height > 0:

        def resize_keep_h(img: Image) -> Image:
            h, w = img.shape[:2]
            if h == target_height:
                return img
            scale = target_height / float(max(1, h))
            return cv2.resize(img, (int(round(w * scale)), int(target_height)), interpolation=cv2.INTER_AREA)

        sides = [resize_keep_h(side) for side in sides]
    return np.concatenate(sides, axis=1)


def draw_heatmap(
    matrix: np.ndarray,
    row_labels: Iterable[str | int],
    col_labels: Iterable[str | int],
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cell_height: int = 80,
    cell_width: int = 80,
) -> Image:
    m = np.array(matrix, dtype=np.float32)
    if m.size == 0:
        return np.full((120, 240, 3), 30, dtype=np.uint8)
    if vmin is None or vmax is None:
        m_min = float(np.nanmin(m)) if np.isfinite(m).any() else 0.0
        m_max = float(np.nanmax(m)) if np.isfinite(m).any() else 1.0
    else:
        m_min = float(vmin)
        m_max = float(vmax)
    m = np.nan_to_num(m, nan=m_max)
    m_clipped = np.clip(m, m_min, m_max)
    denom = (m_max - m_min) if (m_max - m_min) >= 1e-6 else 1.0
    norm = ((m_clipped - m_min) / denom * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_AUTUMN)
    hm = cv2.resize(heat, (heat.shape[1] * cell_width, heat.shape[0] * cell_height), interpolation=cv2.INTER_AREA)
    top_margin, left_margin = 50, 36
    colorbar_w, colorbar_gap = 16, 8
    canvas_h = hm.shape[0] + top_margin
    canvas_w = hm.shape[1] + left_margin + colorbar_gap + colorbar_w
    canvas = np.full((canvas_h, canvas_w, 3), 15, dtype=np.uint8)
    canvas[top_margin:, left_margin : left_margin + hm.shape[1]] = hm
    rows, cols = m.shape
    for r in range(rows + 1):
        y = top_margin + r * cell_height
        cv2.line(canvas, (left_margin, y), (left_margin + cols * cell_width, y), (60, 60, 60), 1)
    for c in range(cols + 1):
        x = left_margin + c * cell_width
        cv2.line(canvas, (x, top_margin), (x, top_margin + rows * cell_height), (60, 60, 60), 1)
    draw_text(canvas, title, (6, 16), (230, 230, 230), 0.45, 1)
    # Labels
    for r, lbl in enumerate(list(row_labels)):
        y = top_margin + r * cell_height + int(cell_height * 0.7)
        draw_text(canvas, f'Track {lbl}', (4, y), (200, 200, 200), 0.35, 1)
    for c, lbl in enumerate(list(col_labels)):
        x = left_margin + c * cell_width + 2
        draw_text(canvas, f'Det {lbl}', (x, top_margin - 6), (200, 200, 200), 0.35, 1)
    # Colorbar
    bar_h = hm.shape[0]
    grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_AUTUMN)
    x0 = left_margin + hm.shape[1] + colorbar_gap
    canvas[top_margin:, x0 : x0 + colorbar_w] = cv2.resize(
        grad_color, (colorbar_w, bar_h), interpolation=cv2.INTER_AREA
    )
    # Annotate each cell with its numeric value
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows, cols = m.shape
    for r in range(rows):
        for c in range(cols):
            val = float(m[r, c])
            label = f'{val:.2f}'
            (tw, th), _ = cv2.getTextSize(label, font, 0.35, 1)
            cx = left_margin + c * cell_width + cell_width // 2
            cy = top_margin + r * cell_height + cell_height // 2 + th // 2
            cv2.putText(canvas, label, (cx - tw // 2, cy), font, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (cx - tw // 2, cy), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
