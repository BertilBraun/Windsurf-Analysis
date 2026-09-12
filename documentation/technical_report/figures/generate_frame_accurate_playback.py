from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT_DIRECTORY = Path(__file__).resolve().parent

BACKGROUND = '#F8F9FB'
INK = '#181F27'
MUTED = '#5B6774'
GRID = '#D3D9E0'
LIGHT = '#EDF0F3'
BLUE = '#126E82'
BLUE_LIGHT = '#DCECEF'
ORANGE = '#E67E22'
ORANGE_LIGHT = '#F8E8D7'
RED = '#C0392B'
WHITE = '#FFFFFF'


@dataclass(frozen=True)
class Packet:
    frame_index: int
    presentation_timestamp: float
    sequence_number: int
    is_keyframe: bool = False


PACKETS = (
    Packet(0, 0.000, 0, True),
    Packet(1, 0.033, 1),
    Packet(2, 0.067, 2),
    Packet(3, 0.067, 3),
    Packet(4, 0.100, 4, True),
    Packet(5, 0.133, 5),
    Packet(6, 0.167, 6),
    Packet(7, 0.200, 7),
    Packet(8, 0.233, 8),
    Packet(9, 0.267, 9),
)


def add_box(
    axes: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = GRID,
    linewidth: float = 1.0,
    radius: float = 0.08,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f'round,pad=0.02,rounding_size={radius}',
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    axes.add_patch(box)
    return box


def add_arrow(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.4,
    mutation_scale: float = 11,
) -> None:
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle='-|>',
            color=color,
            linewidth=linewidth,
            mutation_scale=mutation_scale,
            shrinkA=0,
            shrinkB=0,
        )
    )


def add_section_label(axes: Axes, x: float, y: float, number: str, title: str) -> None:
    axes.text(x, y, number, color=BLUE, fontsize=8.5, fontweight='bold', va='center')
    axes.text(x + 0.25, y, title, color=INK, fontsize=10.5, fontweight='bold', va='center')


def draw_packet_index(axes: Axes) -> None:
    add_section_label(axes, 0.25, 4.25, '1', 'Index packets by presentation order')
    axes.text(
        0.25,
        3.96,
        'sort key: presentation timestamp (PTS), then packet sequence',
        color=MUTED,
        fontsize=8,
    )

    packet_width = 0.58
    packet_height = 0.72
    gap = 0.05
    start_x = 0.25
    y = 3.02
    for packet in PACKETS:
        x = start_x + packet.frame_index * (packet_width + gap)
        fill = ORANGE_LIGHT if packet.is_keyframe else WHITE
        edge = ORANGE if packet.is_keyframe else GRID
        axes.add_patch(Rectangle((x, y), packet_width, packet_height, facecolor=fill, edgecolor=edge, linewidth=1.2))
        axes.text(
            x + packet_width / 2,
            y + 0.48,
            f'F{packet.frame_index}',
            color=INK,
            fontsize=8,
            fontweight='bold',
            ha='center',
        )
        axes.text(
            x + packet_width / 2,
            y + 0.23,
            f'{packet.presentation_timestamp:.3f}s',
            color=MUTED,
            fontsize=6.5,
            ha='center',
        )
        if packet.is_keyframe:
            axes.text(x + 0.04, y + 0.60, 'K', color=ORANGE, fontsize=6.5, fontweight='bold')

    duplicate_x = start_x + 2.5 * (packet_width + gap)
    axes.annotate(
        'equal PTS\nordered by sequence',
        xy=(duplicate_x, y - 0.02),
        xytext=(duplicate_x, y - 0.42),
        color=MUTED,
        fontsize=6.8,
        ha='center',
        va='top',
        arrowprops={'arrowstyle': '-', 'color': GRID, 'linewidth': 1.0},
    )
    axes.text(0.25, 2.72, 'frame index = position in this ordered list', color=BLUE, fontsize=8, fontweight='bold')


def draw_seek_and_cache(axes: Axes) -> None:
    left = 0.25
    add_section_label(axes, left, 2.15, '2', 'Seek from a valid decode boundary')
    axes.text(left, 1.88, 'target F7  ·  requested cache window F5–F9', color=MUTED, fontsize=8)

    y = 1.05
    cell_width = 0.58
    gap = 0.05
    for index in range(4, 10):
        x = left + (index - 4) * (cell_width + gap)
        if index == 7:
            fill, edge, text_color = BLUE, BLUE, WHITE
        elif index == 4:
            fill, edge, text_color = ORANGE_LIGHT, ORANGE, INK
        else:
            fill, edge, text_color = BLUE_LIGHT, BLUE, INK
        axes.add_patch(Rectangle((x, y), cell_width, 0.55, facecolor=fill, edgecolor=edge, linewidth=1.1))
        axes.text(
            x + cell_width / 2, y + 0.29, f'F{index}', color=text_color, fontsize=8, fontweight='bold', ha='center'
        )
        if index == 4:
            axes.text(x + cell_width / 2, y - 0.22, 'restart keyframe', color=ORANGE, fontsize=6.8, ha='center')
        if index == 7:
            axes.text(x + cell_width / 2, y - 0.22, 'display target', color=BLUE, fontsize=6.8, ha='center')

    add_arrow(axes, (left + 0.2, 1.70), (left + 3.64, 1.70), color=BLUE, linewidth=1.8)
    axes.text(left + 1.92, 1.75, 'decode forward', color=BLUE, fontsize=7.2, ha='center')

    add_box(axes, 4.25, 0.80, 2.35, 1.20, facecolor=WHITE)
    axes.text(4.42, 1.75, 'Bounded, race-safe decode', color=INK, fontsize=8.8, fontweight='bold')
    axes.text(4.42, 1.49, 'cache: behind  ←  target  →  ahead', color=MUTED, fontsize=7.0)
    axes.text(4.42, 1.27, 'prefetch; evict outside the window', color=MUTED, fontsize=7.0)
    axes.text(4.42, 1.05, 'op 41 stale  →  discard', color=RED, fontsize=6.9)
    axes.text(4.42, 0.86, 'op 42 current  →  commit', color=BLUE, fontsize=6.9, fontweight='bold')
    add_arrow(axes, (3.95, 1.34), (4.19, 1.34), color=MUTED)


def draw_shared_contract(axes: Axes) -> None:
    add_section_label(axes, 7.15, 4.25, '3', 'Use one frame identity throughout')

    add_box(axes, 7.15, 3.08, 2.02, 0.83, facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.2)
    axes.text(8.16, 3.62, 'Frame contract', color=BLUE, fontsize=9, fontweight='bold', ha='center')
    axes.text(8.16, 3.35, 'frame index + decoded canvas', color=INK, fontsize=7.3, ha='center')
    axes.text(8.16, 3.16, 'packet PTS + duration', color=MUTED, fontsize=6.8, ha='center')

    add_arrow(axes, (8.16, 3.04), (8.16, 2.73), color=BLUE)
    add_arrow(axes, (8.16, 2.73), (7.48, 2.39), color=BLUE)
    add_arrow(axes, (8.16, 2.73), (8.84, 2.39), color=BLUE)

    add_box(axes, 6.72, 1.97, 1.52, 0.42, facecolor=WHITE)
    axes.text(7.48, 2.18, 'Preview', color=INK, fontsize=8.5, fontweight='bold', ha='center', va='center')
    add_box(axes, 8.31, 1.97, 1.52, 0.42, facecolor=WHITE)
    axes.text(9.07, 2.18, 'MP4 export', color=INK, fontsize=8.5, fontweight='bold', ha='center', va='center')

    axes.text(7.48, 1.75, 'overlays at frame Fᵢ', color=MUTED, fontsize=6.8, ha='center')
    axes.text(9.07, 1.75, 'crop at frame Fᵢ', color=MUTED, fontsize=6.8, ha='center')
    axes.text(
        8.27,
        1.39,
        'same frame-index mapping  ·  same detection lookup',
        color=BLUE,
        fontsize=7.2,
        fontweight='bold',
        ha='center',
    )
    axes.text(8.27, 1.10, 'same detailed-crop geometry', color=BLUE, fontsize=7.2, fontweight='bold', ha='center')


def create_figure() -> Figure:
    figure, axes = plt.subplots(figsize=(11.5, 5.0))
    figure.patch.set_facecolor(BACKGROUND)
    axes.set_facecolor(BACKGROUND)
    axes.set_xlim(0, 10)
    axes.set_ylim(0.55, 4.65)
    axes.axis('off')

    axes.text(
        0.25,
        4.57,
        'Frame-accurate browser playback keeps decoding and metadata synchronized',
        color=INK,
        fontsize=13,
        fontweight='bold',
        va='top',
    )

    draw_packet_index(axes)
    draw_seek_and_cache(axes)
    draw_shared_contract(axes)

    figure.tight_layout(pad=0.5)
    return figure


def main() -> None:
    figure = create_figure()
    figure.savefig(
        OUTPUT_DIRECTORY / 'player-frame-accurate-playback.pdf',
        bbox_inches='tight',
        facecolor=figure.get_facecolor(),
    )
    figure.savefig(
        OUTPUT_DIRECTORY / 'player-frame-accurate-playback.png',
        dpi=220,
        bbox_inches='tight',
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


if __name__ == '__main__':
    main()
