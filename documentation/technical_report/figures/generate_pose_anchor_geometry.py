from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


@dataclass(frozen=True)
class PoseAnnotation:
    bounding_box_center: NormalizedPoint
    bounding_box_width: float
    bounding_box_height: float
    boom_mast: NormalizedPoint
    mast_tip: NormalizedPoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIRECTORY = REPOSITORY_ROOT / 'train' / 'detection' / 'datasets' / 'windsurfer_pose'
SOURCE_STEM = 'edge_cases_MVI_2411_sample_0087'
SOURCE_IMAGE = DATASET_DIRECTORY / 'images' / 'train' / f'{SOURCE_STEM}.jpg'
SOURCE_LABEL = DATASET_DIRECTORY / 'labels' / 'train' / f'{SOURCE_STEM}.txt'
OUTPUT_IMAGE = Path(__file__).with_name('pose-anchor-geometry.png')

# The production anchor interpolates vertically from the box top toward the boom--mast junction.
ANCHOR_INTERPOLATION = 0.85
HORIZONTAL_CROP_MARGIN = 0.07
VERTICAL_CROP_MARGIN = 0.05
OUTPUT_WIDTH = 1400
HEADER_HEIGHT = 110


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = 'arialbd.ttf' if bold else 'arial.ttf'
    return ImageFont.truetype(str(Path('C:/Windows/Fonts') / filename), size=size)


def load_pose_annotation(label_path: Path) -> PoseAnnotation:
    values = [float(value) for value in label_path.read_text(encoding='utf-8').split()]
    if len(values) != 11:
        raise ValueError(f'Expected one YOLO pose row with 11 values in {label_path}')
    if values[7] <= 0 or values[10] <= 0:
        raise ValueError(f'Both pose keypoints must be visible in {label_path}')
    return PoseAnnotation(
        bounding_box_center=NormalizedPoint(values[1], values[2]),
        bounding_box_width=values[3],
        bounding_box_height=values[4],
        boom_mast=NormalizedPoint(values[5], values[6]),
        mast_tip=NormalizedPoint(values[8], values[9]),
    )


def draw_label(
    drawing: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = drawing.textbbox(position, text, font=font)
    drawing.rounded_rectangle(
        (left - 12, top - 7, right + 12, bottom + 7),
        radius=8,
        fill=(12, 18, 24, 215),
    )
    drawing.text(position, text, font=font, fill=color)


def draw_cross(
    drawing: ImageDraw.ImageDraw,
    point: tuple[int, int],
    color: tuple[int, int, int, int],
) -> None:
    x, y = point
    drawing.line((x - 18, y, x + 18, y), fill=color, width=7)
    drawing.line((x, y - 18, x, y + 18), fill=color, width=7)
    drawing.ellipse((x - 3, y - 3, x + 3, y + 3), fill='white')


def draw_marker(
    drawing: ImageDraw.ImageDraw,
    point: tuple[int, int],
    color: tuple[int, int, int, int],
) -> None:
    x, y = point
    drawing.ellipse(
        (x - 12, y - 12, x + 12, y + 12),
        fill=color,
        outline='white',
        width=3,
    )


def create_figure() -> None:
    annotation = load_pose_annotation(SOURCE_LABEL)
    with Image.open(SOURCE_IMAGE) as loaded:
        source = loaded.convert('RGB')
    source_width, source_height = source.size

    box_left = annotation.bounding_box_center.x - annotation.bounding_box_width / 2.0
    box_right = annotation.bounding_box_center.x + annotation.bounding_box_width / 2.0
    box_top = annotation.bounding_box_center.y - annotation.bounding_box_height / 2.0
    box_bottom = annotation.bounding_box_center.y + annotation.bounding_box_height / 2.0
    crop_left = max(0.0, box_left - HORIZONTAL_CROP_MARGIN)
    crop_right = min(1.0, box_right + HORIZONTAL_CROP_MARGIN)
    crop_top = max(0.0, box_top - VERTICAL_CROP_MARGIN)
    crop_bottom = min(1.0, box_bottom + VERTICAL_CROP_MARGIN)
    crop_pixels = (
        round(crop_left * source_width),
        round(crop_top * source_height),
        round(crop_right * source_width),
        round(crop_bottom * source_height),
    )
    crop = source.crop(crop_pixels)
    scale = OUTPUT_WIDTH / crop.width
    image_height = round(crop.height * scale)
    canvas = Image.new('RGB', (OUTPUT_WIDTH, image_height + HEADER_HEIGHT), 'white')
    canvas.paste(
        crop.resize((OUTPUT_WIDTH, image_height), Image.Resampling.LANCZOS),
        (0, HEADER_HEIGHT),
    )

    def to_output(point: NormalizedPoint) -> tuple[int, int]:
        source_x = point.x * source_width
        source_y = point.y * source_height
        return (
            round((source_x - crop_pixels[0]) * scale),
            round((source_y - crop_pixels[1]) * scale + HEADER_HEIGHT),
        )

    bounding_box_center = to_output(annotation.bounding_box_center)
    bounding_box_top_left = to_output(NormalizedPoint(box_left, box_top))
    bounding_box_bottom_right = to_output(NormalizedPoint(box_right, box_bottom))
    boom_mast = to_output(annotation.boom_mast)
    mast_tip = to_output(annotation.mast_tip)
    semantic_anchor = to_output(
        NormalizedPoint(
            annotation.boom_mast.x,
            (1.0 - ANCHOR_INTERPOLATION) * box_top + ANCHOR_INTERPOLATION * annotation.boom_mast.y,
        )
    )

    orange = (242, 153, 74, 255)
    cyan = (78, 205, 220, 255)
    magenta = (235, 92, 155, 255)
    overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(overlay)
    title_font = load_font(40, bold=True)
    label_font = load_font(29, bold=True)

    drawing.text(
        (38, 30),
        'Bounding-box and pose-guided framing geometry',
        font=title_font,
        fill=(22, 28, 35, 255),
    )
    drawing.rectangle(
        (*bounding_box_top_left, *bounding_box_bottom_right),
        outline=orange,
        width=8,
    )
    drawing.line((*mast_tip, *boom_mast), fill=cyan, width=9)
    drawing.line(
        (*bounding_box_center, *semantic_anchor),
        fill=(232, 235, 239, 230),
        width=4,
    )
    draw_cross(drawing, bounding_box_center, orange)
    draw_marker(drawing, mast_tip, cyan)
    draw_marker(drawing, boom_mast, cyan)
    draw_cross(drawing, semantic_anchor, magenta)
    draw_label(
        drawing,
        (bounding_box_top_left[0] + 18, bounding_box_top_left[1] + 16),
        'detector box',
        label_font,
        orange,
    )
    box_center_text = 'box center'
    box_left, box_top_text, _, box_bottom = drawing.textbbox((0, 0), box_center_text, font=label_font)
    box_center_label = (
        bounding_box_center[0] + 36 - box_left,
        round(bounding_box_center[1] - (box_top_text + box_bottom) / 2),
    )
    draw_label(drawing, box_center_label, box_center_text, label_font, orange)
    draw_label(
        drawing,
        (mast_tip[0] + 30, mast_tip[1] - 20),
        'mast tip',
        label_font,
        cyan,
    )
    draw_label(
        drawing,
        (boom_mast[0] + 35, boom_mast[1] - 12),
        'boom-mast junction',
        label_font,
        cyan,
    )
    semantic_anchor_text = 'semantic anchor'
    _, semantic_top, semantic_right, semantic_bottom = drawing.textbbox((0, 0), semantic_anchor_text, font=label_font)
    semantic_anchor_label = (
        semantic_anchor[0] - 36 - semantic_right,
        round(semantic_anchor[1] - (semantic_top + semantic_bottom) / 2),
    )
    draw_label(
        drawing,
        semantic_anchor_label,
        semantic_anchor_text,
        label_font,
        magenta,
    )

    composed = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
    composed.save(OUTPUT_IMAGE, optimize=True, dpi=(300, 300))


if __name__ == '__main__':
    create_figure()
