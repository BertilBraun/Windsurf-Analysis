from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIRECTORY = Path(__file__).resolve().parent

BACKGROUND = '#F8F9FB'
INK = '#181F27'
MUTED = '#5B6774'
GRID = '#D3D9E0'
BLUE = '#126E82'
BLUE_LIGHT = '#DCECEF'
ORANGE = '#E67E22'
ORANGE_LIGHT = '#F8E8D7'
WHITE = '#FFFFFF'


def add_box(
    axes: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = GRID,
    linewidth: float = 1.2,
    radius: float = 0.08,
) -> None:
    axes.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f'round,pad=0.02,rounding_size={radius}',
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )


def add_arrow(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BLUE,
    linewidth: float = 1.6,
) -> None:
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle='-|>',
            color=color,
            linewidth=linewidth,
            mutation_scale=12,
            shrinkA=0,
            shrinkB=0,
        )
    )


def add_step(
    axes: Axes,
    x: float,
    y: float,
    number: str,
    title: str,
    detail: str,
    *,
    accent: str,
) -> None:
    axes.text(x, y, number, color=accent, fontsize=8.3, fontweight='bold', va='top')
    axes.text(x + 0.28, y, title, color=INK, fontsize=9.0, fontweight='bold', va='top')
    axes.text(x + 0.28, y - 0.26, detail, color=MUTED, fontsize=7.6, va='top', linespacing=1.25)


def draw_canonical_order(axes: Axes) -> None:
    add_box(axes, 0.52, 3.92, 8.96, 1.08, facecolor=WHITE)
    axes.text(0.78, 4.73, 'Canonical frame identity', color=INK, fontsize=10.2, fontweight='bold')
    axes.text(
        0.78,
        4.43,
        'Discard negative PTS; sort presentable packets by PTS, then sequence number.',
        color=MUTED,
        fontsize=8.2,
    )

    packet_labels = ('i−1', 'i', 'i+1')
    for position, label in enumerate(packet_labels):
        x = 5.86 + position * 0.82
        selected = position == 1
        add_box(
            axes,
            x,
            4.18,
            0.64,
            0.48,
            facecolor=BLUE if selected else BLUE_LIGHT,
            edgecolor=BLUE,
            linewidth=1.3 if selected else 0.9,
            radius=0.05,
        )
        axes.text(
            x + 0.32,
            4.42,
            label,
            color=WHITE if selected else BLUE,
            fontsize=9.0,
            fontweight='bold',
            ha='center',
            va='center',
        )

    axes.text(8.48, 4.56, 'frame i', color=BLUE, fontsize=9.6, fontweight='bold', ha='center')
    axes.text(8.48, 4.28, '= position in this order', color=MUTED, fontsize=7.5, ha='center')

    add_arrow(axes, (4.96, 3.88), (2.78, 3.50))
    add_arrow(axes, (5.04, 3.88), (7.22, 3.50))


def draw_preview_lane(axes: Axes) -> None:
    add_box(axes, 0.52, 0.78, 4.26, 2.70, facecolor=WHITE, edgecolor=BLUE, linewidth=1.3)
    axes.text(0.78, 3.18, 'Interactive preview', color=BLUE, fontsize=11.0, fontweight='bold')
    axes.text(4.50, 3.18, 'independent decoder', color=MUTED, fontsize=7.4, ha='right')

    add_step(
        axes,
        0.80,
        2.80,
        '1',
        'Decode packet i',
        'On a cache miss: preceding keyframe → decode forward.',
        accent=BLUE,
    )
    add_step(
        axes,
        0.80,
        2.12,
        '2',
        'Look up metadata at i',
        'Detection for overlays/crop; stabilization[i] for overview.',
        accent=BLUE,
    )
    add_step(
        axes,
        0.80,
        1.44,
        '3',
        'Render the selected view',
        'Overview, or a focused crop from pose anchor + scale.',
        accent=BLUE,
    )


def draw_export_lane(axes: Axes) -> None:
    add_box(axes, 5.22, 0.78, 4.26, 2.70, facecolor=WHITE, edgecolor=ORANGE, linewidth=1.3)
    axes.text(5.48, 3.18, 'Focused video export', color=ORANGE, fontsize=11.0, fontweight='bold')
    axes.text(9.20, 3.18, 'independent decoder', color=MUTED, fontsize=7.4, ha='right')

    add_step(
        axes,
        5.50,
        2.80,
        '1',
        'Recover packet index i',
        'Advance in order; use sample PTS to resynchronize if needed.',
        accent=ORANGE,
    )
    add_step(
        axes,
        5.50,
        2.12,
        '2',
        'Look up the track at i',
        'Select the closest detection for the requested rider.',
        accent=ORANGE,
    )
    add_step(
        axes,
        5.50,
        1.44,
        '3',
        'Render the exported frame',
        'Apply the same focused-crop function: anchor + scale.',
        accent=ORANGE,
    )


def draw_shared_contract(axes: Axes) -> None:
    axes.plot([2.22, 2.22, 7.76, 7.76], [0.68, 0.54, 0.54, 0.68], color=BLUE, linewidth=1.4)
    axes.text(
        4.99,
        0.28,
        'Shared: ordered-packet index and focused-crop function  ·  Separate: decoding and output surfaces',
        color=BLUE,
        fontsize=8.2,
        fontweight='bold',
        ha='center',
    )


def create_figure() -> Figure:
    figure, axes = plt.subplots(figsize=(11.5, 6.0))
    figure.patch.set_facecolor(BACKGROUND)
    axes.set_facecolor(BACKGROUND)
    axes.set_xlim(0, 10)
    axes.set_ylim(0.10, 5.70)
    axes.axis('off')

    axes.text(
        0.50,
        5.60,
        'One packet-order index keeps preview and export aligned',
        color=INK,
        fontsize=14,
        fontweight='bold',
        va='top',
    )
    axes.text(
        0.50,
        5.30,
        'Both paths identify frame i the same way, then decode and render it independently.',
        color=MUTED,
        fontsize=9,
        va='top',
    )

    draw_canonical_order(axes)
    draw_preview_lane(axes)
    draw_export_lane(axes)
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
