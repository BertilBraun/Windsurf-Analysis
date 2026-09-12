from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class NormalizedBoundingBox:
    center_x: float
    center_y: float
    width: float
    height: float


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORY = REPOSITORY_ROOT / 'train' / 'detection' / 'windsurf_dataset'
OUTPUT_DIRECTORY = Path(__file__).resolve().parent
SOURCE_FRAMES = tuple(
    DATASET_DIRECTORY / f'S_2025_07_15_S_C_B_R_001_sample_{sample:04d}.jpg' for sample in (100, 101, 102)
)
SOURCE_BOXES = (
    (
        NormalizedBoundingBox(0.237274, 0.588942, 0.036911, 0.143120),
        NormalizedBoundingBox(0.761719, 0.614489, 0.042188, 0.147163),
    ),
    (
        NormalizedBoundingBox(0.228179, 0.590110, 0.040517, 0.141772),
        NormalizedBoundingBox(0.750615, 0.609722, 0.041854, 0.149074),
    ),
    (
        NormalizedBoundingBox(0.198437, 0.593725, 0.035417, 0.147279),
        NormalizedBoundingBox(0.716892, 0.609740, 0.040834, 0.137926),
    ),
)

BACKGROUND = (248, 249, 251)
INK = (24, 31, 39)
MUTED = (91, 103, 116)
GRID = (211, 217, 224)
BLUE = (18, 110, 130)
ORANGE = (230, 126, 34)
RED = (192, 57, 43)
WHITE = (255, 255, 255)


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = 'arialbd.ttf' if bold else 'arial.ttf'
    return ImageFont.truetype(str(Path('C:/Windows/Fonts') / filename), size=size)


def centered_text(
    drawing: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = drawing.textbbox((0, 0), text, font=font)
    width = right - left
    height = bottom - top
    x = box[0] + (box[2] - box[0] - width) // 2
    y = box[1] + (box[3] - box[1] - height) // 2 - top
    drawing.text((x, y), text, font=font, fill=fill)


def crop_detection(image: Image.Image, bounding_box: NormalizedBoundingBox) -> Image.Image:
    width, height = image.size
    padding = 0.20
    half_width = bounding_box.width * width * (0.5 + padding)
    half_height = bounding_box.height * height * (0.5 + padding)
    left = max(0, int(round(bounding_box.center_x * width - half_width)))
    right = min(width, int(round(bounding_box.center_x * width + half_width)))
    top = max(0, int(round(bounding_box.center_y * height - half_height)))
    bottom = min(height, int(round(bounding_box.center_y * height + half_height)))
    return image.crop((left, top, right, bottom))


def fit_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (int(round(image.width * scale)), int(round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new('RGB', (width, height), WHITE)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def compute_similarity(crops: list[Image.Image]) -> np.ndarray:
    sys.path.insert(0, str(REPOSITORY_ROOT / 'video_processing'))
    from inference.src.tracking.reid.ReIDColorABStripeHistogram import (  # noqa: PLC0415
        ReIDColorABStripeHistogram,
    )

    descriptor = ReIDColorABStripeHistogram()
    bgr_crops = [cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR) for crop in crops]
    embeddings = descriptor.get_features_for_crops(bgr_crops)
    identity_indices = ((0, 2, 4), (1, 3, 5))
    similarity = np.zeros((2, 2), dtype=np.float32)
    for row, row_indices in enumerate(identity_indices):
        for column, column_indices in enumerate(identity_indices):
            values = [
                1.0 - embeddings[first].distance(embeddings[second])
                for first in row_indices
                for second in column_indices
                if row != column or first != second
            ]
            similarity[row, column] = float(np.mean(values))
    return similarity


def similarity_color(value: float) -> tuple[int, int, int]:
    clipped = max(0.0, min(1.0, value))
    low = np.asarray((241, 218, 207), dtype=np.float32)
    high = np.asarray((84, 166, 151), dtype=np.float32)
    color = np.round(low * (1.0 - clipped) + high * clipped).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def create_sail_appearance_figure() -> None:
    frames = [Image.open(path).convert('RGB') for path in SOURCE_FRAMES]
    crops = [
        crop_detection(frame, bounding_box)
        for frame, frame_boxes in zip(frames, SOURCE_BOXES, strict=True)
        for bounding_box in frame_boxes
    ]
    similarity = compute_similarity(crops)

    canvas = Image.new('RGB', (1500, 820), BACKGROUND)
    drawing = ImageDraw.Draw(canvas)
    title_font = load_font(38, bold=True)
    heading_font = load_font(25, bold=True)
    label_font = load_font(23)
    score_font = load_font(31, bold=True)
    note_font = load_font(20)

    drawing.text((55, 34), 'Sail color remains discriminative across time', font=title_font, fill=INK)
    drawing.text(
        (55, 88),
        'Three detections per rider from consecutive samples of the same project sequence',
        font=note_font,
        fill=MUTED,
    )

    crop_width = 190
    crop_height = 255
    x_positions = (180, 400, 620)
    y_positions = (165, 475)
    identity_names = ('Track A', 'Track B')
    identity_colors = (BLUE, ORANGE)
    for identity_index in range(2):
        drawing.text(
            (55, y_positions[identity_index] + 102),
            identity_names[identity_index],
            font=heading_font,
            fill=identity_colors[identity_index],
        )
        for time_index, x_position in enumerate(x_positions):
            crop_index = time_index * 2 + identity_index
            fitted = fit_crop(crops[crop_index], crop_width, crop_height)
            canvas.paste(fitted, (x_position, y_positions[identity_index]))
            drawing.rectangle(
                (
                    x_position,
                    y_positions[identity_index],
                    x_position + crop_width,
                    y_positions[identity_index] + crop_height,
                ),
                outline=identity_colors[identity_index],
                width=5,
            )
            if identity_index == 0:
                centered_text(
                    drawing,
                    (x_position, 130, x_position + crop_width, 160),
                    f't{time_index + 1}',
                    label_font,
                    MUTED,
                )

    matrix_left = 985
    matrix_top = 265
    cell_size = 165
    drawing.text((930, 165), 'Descriptor similarity', font=heading_font, fill=INK)
    drawing.text((930, 205), 'production color embedding', font=note_font, fill=MUTED)
    for index, label in enumerate(('A', 'B')):
        centered_text(
            drawing,
            (matrix_left + index * cell_size, matrix_top - 48, matrix_left + (index + 1) * cell_size, matrix_top),
            label,
            heading_font,
            INK,
        )
        centered_text(
            drawing,
            (matrix_left - 55, matrix_top + index * cell_size, matrix_left, matrix_top + (index + 1) * cell_size),
            label,
            heading_font,
            INK,
        )
    for row in range(2):
        for column in range(2):
            box = (
                matrix_left + column * cell_size,
                matrix_top + row * cell_size,
                matrix_left + (column + 1) * cell_size,
                matrix_top + (row + 1) * cell_size,
            )
            drawing.rectangle(box, fill=similarity_color(float(similarity[row, column])), outline=WHITE, width=5)
            centered_text(drawing, box, f'{similarity[row, column]:.2f}', score_font, INK)

    drawing.text((930, 625), 'higher = more similar', font=note_font, fill=MUTED)
    drawing.text(
        (930, 675),
        'Lab chromaticity + circular hue',
        font=note_font,
        fill=INK,
    )
    drawing.text((930, 706), 'saturation-weighted; 3 vertical stripes', font=note_font, fill=INK)
    drawing.text((930, 737), '+ one global histogram', font=note_font, fill=INK)

    canvas.save(OUTPUT_DIRECTORY / 'tracking-sail-color-similarity.png', optimize=True, dpi=(300, 300))


def arrow(
    drawing: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    drawing.line((*start, *end), fill=color, width=width)
    direction = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    direction /= np.linalg.norm(direction)
    perpendicular = np.asarray((-direction[1], direction[0]), dtype=np.float32)
    tip = np.asarray(end, dtype=np.float32)
    base = tip - direction * 18
    points = [tip, base + perpendicular * 8, base - perpendicular * 8]
    drawing.polygon([(int(point[0]), int(point[1])) for point in points], fill=color)


def create_offline_association_figure() -> None:
    canvas = Image.new('RGB', (1500, 760), BACKGROUND)
    drawing = ImageDraw.Draw(canvas)
    title_font = load_font(38, bold=True)
    heading_font = load_font(25, bold=True)
    small_font = load_font(19)

    drawing.text((55, 34), 'Offline global association resolves ambiguous gaps', font=title_font, fill=INK)
    drawing.text(
        (55, 88),
        'Reliable local links form tracklets; the full-video optimizer selects a consistent set of continuations.',
        font=small_font,
        fill=MUTED,
    )

    timeline_y = 675
    arrow(drawing, (105, timeline_y), (1410, timeline_y), MUTED, 3)
    drawing.text((1300, 690), 'video time', font=small_font, fill=MUTED)

    drawing.rounded_rectangle((565, 145, 825, 620), radius=18, fill=(235, 238, 242), outline=GRID, width=2)
    centered_text(drawing, (565, 155, 825, 195), 'long overlap / occlusion', small_font, MUTED)

    tracklets = (
        ('A1', 125, 390, 260, BLUE),
        ('B1', 125, 470, 390, ORANGE),
        ('A2', 850, 1135, 260, BLUE),
        ('B2', 850, 1135, 390, ORANGE),
        ('A3', 1190, 1390, 260, BLUE),
        ('B3', 1190, 1390, 390, ORANGE),
    )
    for label, left, right, y_position, color in tracklets:
        drawing.rounded_rectangle((left, y_position - 30, right, y_position + 30), radius=18, fill=color)
        centered_text(drawing, (left, y_position - 30, right, y_position + 30), label, heading_font, WHITE)

    candidate = (151, 158, 166)
    arrow(drawing, (470, 250), (840, 370), candidate, 4)
    arrow(drawing, (470, 390), (840, 280), candidate, 4)
    drawing.rounded_rectangle((515, 282, 680, 319), radius=8, fill=BACKGROUND)
    centered_text(drawing, (515, 282, 680, 319), 'plausible locally', small_font, MUTED)
    drawing.rounded_rectangle((674, 335, 839, 372), radius=8, fill=BACKGROUND)
    centered_text(drawing, (674, 335, 839, 372), 'plausible locally', small_font, MUTED)

    arrow(drawing, (470, 250), (840, 250), BLUE, 8)
    arrow(drawing, (470, 390), (840, 390), ORANGE, 8)
    arrow(drawing, (1135, 250), (1180, 250), BLUE, 8)
    arrow(drawing, (1135, 390), (1180, 390), ORANGE, 8)

    drawing.text((120, 165), 'high-purity tracklets', font=heading_font, fill=INK)
    drawing.text((910, 165), 'future evidence constrains the choice', font=heading_font, fill=INK)

    legend_y = 555
    drawing.line((950, legend_y, 1020, legend_y), fill=BLUE, width=8)
    drawing.text((1040, legend_y - 13), 'selected continuation', font=small_font, fill=INK)
    drawing.line((950, legend_y + 48, 1020, legend_y + 48), fill=candidate, width=4)
    drawing.text((1040, legend_y + 35), 'rejected candidate', font=small_font, fill=INK)

    drawing.rounded_rectangle((90, 500, 730, 615), radius=16, fill=WHITE, outline=GRID, width=2)
    drawing.text((120, 520), 'Edge evidence', font=heading_font, fill=INK)
    drawing.text(
        (120, 562),
        'camera-compensated motion  +  sail appearance  +  gap duration',
        font=small_font,
        fill=MUTED,
    )

    canvas.save(OUTPUT_DIRECTORY / 'tracking-offline-global-association.png', optimize=True, dpi=(300, 300))


def main() -> None:
    create_sail_appearance_figure()
    create_offline_association_figure()


if __name__ == '__main__':
    main()
