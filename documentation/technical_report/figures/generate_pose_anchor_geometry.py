from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


@dataclass(frozen=True)
class NormalizedBoundingBox:
    center: NormalizedPoint
    width: float
    height: float


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_IMAGE = REPOSITORY_ROOT / 'train' / 'detection' / 'windsurf_dataset' / 'P1030704_frame_000765.jpg'
OUTPUT_IMAGE = Path(__file__).with_name('pose-anchor-geometry.png')

BOUNDING_BOX = NormalizedBoundingBox(
    center=NormalizedPoint(x=0.473629, y=0.407902),
    width=0.191827,
    height=0.794255,
)
BOOM_MAST = NormalizedPoint(x=0.471199, y=0.561913)
MAST_TIP = NormalizedPoint(x=0.562497, y=0.010648)

# The production anchor lies 85% of the way from a proxy mast-tip position
# at the box top toward the boom/mast junction.
ANCHOR_INTERPOLATION = 0.85
CROP_LEFT = 1100
CROP_TOP = 0
CROP_RIGHT = 2530
CROP_BOTTOM = 1900
OUTPUT_WIDTH = 1200


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = 'arialbd.ttf' if bold else 'arial.ttf'
    return ImageFont.truetype(str(Path('C:/Windows/Fonts') / filename), size=size)


def to_source_pixels(point: NormalizedPoint, width: int, height: int) -> tuple[float, float]:
    return point.x * width, point.y * height


def to_output_pixels(
    point: tuple[float, float],
    scale: float,
    header_height: int,
) -> tuple[int, int]:
    x = int(round((point[0] - CROP_LEFT) * scale))
    y = int(round((point[1] - CROP_TOP) * scale + header_height))
    return x, y


def draw_marker(
    drawing: ImageDraw.ImageDraw,
    point: tuple[int, int],
    color: tuple[int, int, int, int],
    radius: int,
) -> None:
    x, y = point
    drawing.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline='white', width=3)


def draw_cross(
    drawing: ImageDraw.ImageDraw,
    point: tuple[int, int],
    color: tuple[int, int, int, int],
    radius: int,
) -> None:
    x, y = point
    drawing.line((x - radius, y, x + radius, y), fill=color, width=7)
    drawing.line((x, y - radius, x, y + radius), fill=color, width=7)
    drawing.ellipse((x - 3, y - 3, x + 3, y + 3), fill='white')


def draw_label(
    drawing: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int, int],
) -> None:
    x, y = position
    left, top, right, bottom = drawing.textbbox((x, y), text, font=font)
    padding_x = 12
    padding_y = 7
    drawing.rounded_rectangle(
        (left - padding_x, top - padding_y, right + padding_x, bottom + padding_y),
        radius=8,
        fill=(12, 18, 24, 215),
    )
    drawing.text((x, y), text, font=font, fill=color)


def create_figure() -> None:
    source = Image.open(SOURCE_IMAGE).convert('RGB')
    source_width, source_height = source.size

    crop = source.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM))
    scale = OUTPUT_WIDTH / crop.width
    image_height = int(round(crop.height * scale))
    header_height = 100
    canvas = Image.new('RGB', (OUTPUT_WIDTH, image_height + header_height), color=(246, 247, 249))
    canvas.paste(crop.resize((OUTPUT_WIDTH, image_height), Image.Resampling.LANCZOS), (0, header_height))

    overlay = Image.new('RGBA', canvas.size, color=(0, 0, 0, 0))
    drawing = ImageDraw.Draw(overlay)
    title_font = load_font(35, bold=True)
    label_font = load_font(25, bold=True)
    small_font = load_font(22)

    drawing.text(
        (34, 26),
        'Bounding-box geometry versus semantic pose geometry',
        font=title_font,
        fill=(22, 28, 35, 255),
    )

    center_source = to_source_pixels(BOUNDING_BOX.center, source_width, source_height)
    half_width = BOUNDING_BOX.width * source_width / 2.0
    half_height = BOUNDING_BOX.height * source_height / 2.0
    box_top_source = center_source[1] - half_height
    bbox_top_left = to_output_pixels((center_source[0] - half_width, box_top_source), scale, header_height)
    bbox_bottom_right = to_output_pixels(
        (center_source[0] + half_width, center_source[1] + half_height),
        scale,
        header_height,
    )

    boom_source = to_source_pixels(BOOM_MAST, source_width, source_height)
    mast_tip_source = to_source_pixels(MAST_TIP, source_width, source_height)
    anchor_source = (
        boom_source[0],
        (1.0 - ANCHOR_INTERPOLATION) * box_top_source + ANCHOR_INTERPOLATION * boom_source[1],
    )

    bbox_center = to_output_pixels(center_source, scale, header_height)
    boom = to_output_pixels(boom_source, scale, header_height)
    mast_tip = to_output_pixels(mast_tip_source, scale, header_height)
    anchor = to_output_pixels(anchor_source, scale, header_height)

    orange = (242, 153, 74, 255)
    cyan = (78, 205, 220, 255)
    magenta = (235, 92, 155, 255)

    drawing.rectangle((*bbox_top_left, *bbox_bottom_right), outline=orange, width=7)
    drawing.line((*mast_tip, *boom), fill=cyan, width=8)
    drawing.line((*bbox_center, *anchor), fill=(232, 235, 239, 230), width=4)
    draw_cross(drawing, bbox_center, orange, radius=18)
    draw_marker(drawing, mast_tip, cyan, radius=12)
    draw_marker(drawing, boom, cyan, radius=12)
    draw_cross(drawing, anchor, magenta, radius=18)

    draw_label(
        drawing,
        (bbox_top_left[0] + 18, bbox_top_left[1] + 16),
        'detector box',
        label_font,
        orange,
    )
    draw_label(
        drawing,
        (bbox_center[0] - 250, bbox_center[1] - 30),
        'box center',
        label_font,
        orange,
    )
    draw_label(
        drawing,
        (mast_tip[0] - 220, mast_tip[1] + 24),
        'mast tip',
        label_font,
        cyan,
    )
    draw_label(
        drawing,
        (boom[0] - 315, boom[1] + 8),
        'boom--mast junction',
        label_font,
        cyan,
    )
    draw_label(
        drawing,
        (anchor[0] + 35, anchor[1] - 25),
        'semantic anchor',
        label_font,
        magenta,
    )
    draw_label(
        drawing,
        (mast_tip[0] + 25, (mast_tip[1] + boom[1]) // 2),
        'mast-length scale signal',
        small_font,
        cyan,
    )

    composed = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
    composed.save(OUTPUT_IMAGE, optimize=True, dpi=(300, 300))


if __name__ == '__main__':
    create_figure()
