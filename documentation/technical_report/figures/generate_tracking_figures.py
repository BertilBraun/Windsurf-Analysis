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


@dataclass(frozen=True)
class AppearanceMatchScores:
    within_a: float
    within_b: float
    between_a_and_b: float


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
PRODUCTION_APPEARANCE_SIMILARITY_GAMMA = 11.630051976558498


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


def compute_appearance_match_scores(crops: list[Image.Image]) -> AppearanceMatchScores:
    sys.path.insert(0, str(REPOSITORY_ROOT / 'video_processing'))
    from inference.src.tracking.reid.ReIDColorABStripeHistogram import (  # noqa: PLC0415
        ReIDColorABStripeHistogram,
    )

    descriptor = ReIDColorABStripeHistogram()
    bgr_crops = [cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR) for crop in crops]
    embeddings = descriptor.get_features_for_crops(bgr_crops)
    identity_indices = ((0, 2, 4), (1, 3, 5))
    match_scores = np.zeros((2, 2), dtype=np.float32)
    for row, row_indices in enumerate(identity_indices):
        for column, column_indices in enumerate(identity_indices):
            values = [
                embeddings[first].probability(embeddings[second], PRODUCTION_APPEARANCE_SIMILARITY_GAMMA)
                for first in row_indices
                for second in column_indices
                if row != column or first != second
            ]
            match_scores[row, column] = float(np.mean(values))
    return AppearanceMatchScores(
        within_a=float(match_scores[0, 0]),
        within_b=float(match_scores[1, 1]),
        between_a_and_b=float((match_scores[0, 1] + match_scores[1, 0]) / 2.0),
    )


def match_score_color(value: float) -> tuple[int, int, int]:
    clipped = max(0.0, min(1.0, value))
    low = np.asarray((241, 218, 207), dtype=np.float32)
    high = np.asarray((84, 166, 151), dtype=np.float32)
    color = np.round(low * (1.0 - clipped) + high * clipped).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def format_match_score(value: float) -> str:
    return '<0.001' if value < 0.001 else f'{value:.2f}'


def create_sail_appearance_figure() -> None:
    frames = [Image.open(path).convert('RGB') for path in SOURCE_FRAMES]
    crops = [
        crop_detection(frame, bounding_box)
        for frame, frame_boxes in zip(frames, SOURCE_BOXES, strict=True)
        for bounding_box in frame_boxes
    ]
    match_scores = compute_appearance_match_scores(crops)

    canvas = Image.new('RGB', (1500, 820), BACKGROUND)
    drawing = ImageDraw.Draw(canvas)
    title_font = load_font(38, bold=True)
    heading_font = load_font(25, bold=True)
    label_font = load_font(23)
    score_font = load_font(29, bold=True)
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

    matrix_left = 1010
    matrix_top = 265
    cell_size = 165
    drawing.text((930, 165), 'Appearance match score', font=heading_font, fill=INK)
    drawing.text((930, 205), 'production mapping; higher = stronger match', font=note_font, fill=MUTED)
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

    score_matrix = (
        (match_scores.within_a, match_scores.between_a_and_b),
        (match_scores.between_a_and_b, match_scores.within_b),
    )
    for row in range(2):
        for column in range(2):
            value = score_matrix[row][column]
            box = (
                matrix_left + column * cell_size,
                matrix_top + row * cell_size,
                matrix_left + (column + 1) * cell_size,
                matrix_top + (row + 1) * cell_size,
            )
            drawing.rectangle(box, fill=match_score_color(value), outline=WHITE, width=5)
            centered_text(drawing, box, format_match_score(value), score_font, INK)

    drawing.text((930, 625), 'Mean over pairwise comparisons shown', font=note_font, fill=MUTED)
    drawing.text((930, 656), 'Heuristic association score, not accuracy.', font=note_font, fill=MUTED)
    drawing.text(
        (930, 707),
        'Lab chromaticity + circular hue',
        font=note_font,
        fill=INK,
    )
    drawing.text((930, 738), 'saturation-weighted; 3 stripes + global', font=note_font, fill=INK)

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


def dashed_line(
    drawing: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    start_vector = np.asarray(start, dtype=np.float32)
    difference = np.asarray(end, dtype=np.float32) - start_vector
    length = float(np.linalg.norm(difference))
    direction = difference / length
    dash_length = 16.0
    gap_length = 11.0
    position = 0.0
    while position < length:
        segment_end = min(position + dash_length, length)
        first = start_vector + direction * position
        second = start_vector + direction * segment_end
        drawing.line(
            (int(first[0]), int(first[1]), int(second[0]), int(second[1])),
            fill=color,
            width=width,
        )
        position += dash_length + gap_length


def draw_tracklet(
    drawing: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
) -> None:
    drawing.rounded_rectangle(box, radius=16, fill=color)
    centered_text(drawing, box, label, font, WHITE)


def create_offline_association_figure() -> None:
    canvas = Image.new('RGB', (1500, 930), BACKGROUND)
    drawing = ImageDraw.Draw(canvas)
    title_font = load_font(38, bold=True)
    heading_font = load_font(25, bold=True)
    small_font = load_font(19)
    label_font = load_font(21, bold=True)

    drawing.text((55, 34), 'Offline global association resolves ambiguous gaps', font=title_font, fill=INK)
    drawing.text(
        (55, 88),
        'At the gap, both pairings fit. Later observations make one pairing globally consistent.',
        font=small_font,
        fill=MUTED,
    )

    candidate = (151, 158, 166)
    pale_panel = (238, 241, 244)
    gap_fill = (225, 229, 234)
    drawing.rounded_rectangle((55, 135, 1445, 440), radius=18, fill=WHITE, outline=GRID, width=2)
    drawing.text((85, 156), '1  At the gap: a causal decision is ambiguous', font=heading_font, fill=INK)
    drawing.text(
        (85, 197), 'Only past evidence and two short emerging tracklets are visible.', font=small_font, fill=MUTED
    )
    drawing.rounded_rectangle((545, 235, 710, 405), radius=12, fill=gap_fill)
    centered_text(drawing, (545, 245, 710, 280), 'association gap', small_font, MUTED)
    drawing.rectangle((1040, 225, 1410, 415), fill=pale_panel)
    drawing.line((1040, 220, 1040, 420), fill=MUTED, width=3)
    drawing.text((1013, 425), 'now', font=small_font, fill=MUTED)
    centered_text(drawing, (1070, 290, 1380, 350), 'future not yet observed', heading_font, MUTED)

    top_a = (150, 260, 500, 310)
    top_b = (150, 350, 500, 400)
    top_c = (755, 260, 965, 310)
    top_d = (755, 350, 965, 400)
    draw_tracklet(drawing, top_a, 'Tracklet A', BLUE, label_font)
    draw_tracklet(drawing, top_b, 'Tracklet B', ORANGE, label_font)
    draw_tracklet(drawing, top_c, 'Tracklet C', candidate, label_font)
    draw_tracklet(drawing, top_d, 'Tracklet D', candidate, label_font)
    for start_y in (285, 375):
        for end_y in (285, 375):
            dashed_line(drawing, (500, start_y), (755, end_y), candidate, 4)
    centered_text(drawing, (520, 405, 980, 438), 'all four links remain plausible', small_font, MUTED)

    drawing.rounded_rectangle((55, 475, 1445, 845), radius=18, fill=WHITE, outline=GRID, width=2)
    drawing.text((85, 496), '2  Offline: later evidence selects one consistent pairing', font=heading_font, fill=INK)
    drawing.text(
        (85, 537),
        'The optimizer evaluates complete tracklets on both sides of the same gap.',
        font=small_font,
        fill=MUTED,
    )
    drawing.rounded_rectangle((545, 585, 710, 790), radius=12, fill=gap_fill)
    centered_text(drawing, (545, 595, 710, 630), 'association gap', small_font, MUTED)

    bottom_a = (150, 615, 500, 665)
    bottom_b = (150, 720, 500, 770)
    bottom_c = (755, 615, 1040, 665)
    bottom_d = (755, 720, 1040, 770)
    draw_tracklet(drawing, bottom_a, 'Tracklet A', BLUE, label_font)
    draw_tracklet(drawing, bottom_b, 'Tracklet B', ORANGE, label_font)
    draw_tracklet(drawing, bottom_c, 'Tracklet C', ORANGE, label_font)
    draw_tracklet(drawing, bottom_d, 'Tracklet D', BLUE, label_font)

    dashed_line(drawing, (500, 640), (755, 640), candidate, 3)
    dashed_line(drawing, (500, 745), (755, 745), candidate, 3)
    arrow(drawing, (500, 640), (755, 745), BLUE, 8)
    arrow(drawing, (500, 745), (755, 640), ORANGE, 8)

    drawing.rounded_rectangle((1080, 590, 1410, 795), radius=14, fill=pale_panel, outline=GRID, width=2)
    drawing.text((1110, 610), 'Later evidence', font=heading_font, fill=INK)
    drawing.text((1110, 656), 'C matches B', font=label_font, fill=ORANGE)
    drawing.text((1110, 694), 'D matches A', font=label_font, fill=BLUE)
    drawing.text((1110, 744), 'stable sail appearance', font=small_font, fill=MUTED)

    drawing.line((85, 885, 155, 885), fill=BLUE, width=8)
    drawing.text((175, 872), 'selected global link', font=small_font, fill=INK)
    dashed_line(drawing, (425, 885), (495, 885), candidate, 3)
    drawing.text((515, 872), 'rejected candidate', font=small_font, fill=INK)
    arrow(drawing, (1150, 885), (1400, 885), MUTED, 3)
    drawing.text((1025, 872), 'full video timeline', font=small_font, fill=MUTED)

    canvas.save(OUTPUT_DIRECTORY / 'tracking-offline-global-association.png', optimize=True, dpi=(300, 300))


def main() -> None:
    create_sail_appearance_figure()
    create_offline_association_figure()


if __name__ == '__main__':
    main()
