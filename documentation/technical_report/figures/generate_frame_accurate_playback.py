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
    linewidth: float = 1.7,
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


def draw_frame_strip(axes: Axes) -> None:
    axes.text(0.55, 4.37, 'Decoded video', color=INK, fontsize=10, fontweight='bold')
    labels = ('frame i−1', 'frame i', 'frame i+1')
    for position, label in enumerate(labels):
        x = 0.55 + position * 1.18
        selected = position == 1
        add_box(
            axes,
            x,
            3.28,
            0.96,
            0.80,
            facecolor=BLUE_LIGHT if selected else WHITE,
            edgecolor=BLUE if selected else GRID,
            linewidth=1.8 if selected else 1.0,
        )
        axes.text(
            x + 0.48,
            3.68,
            label,
            color=BLUE if selected else MUTED,
            fontsize=8.7,
            fontweight='bold' if selected else 'normal',
            ha='center',
            va='center',
        )
        if selected:
            axes.text(x + 0.48, 3.43, 'pixels', color=MUTED, fontsize=7.4, ha='center')


def draw_metadata(axes: Axes) -> None:
    add_box(axes, 0.55, 1.18, 3.32, 1.25, facecolor=WHITE)
    axes.text(0.78, 2.13, 'Analysis metadata at frame i', color=INK, fontsize=10, fontweight='bold')
    axes.text(0.78, 1.82, '• tracked rider and bounding box', color=MUTED, fontsize=8.5)
    axes.text(0.78, 1.55, '• pose anchor and crop scale', color=MUTED, fontsize=8.5)
    axes.text(0.78, 1.28, '• camera-stabilization transform', color=MUTED, fontsize=8.5)


def draw_contract(axes: Axes) -> None:
    add_box(axes, 4.52, 2.12, 1.62, 1.22, facecolor=BLUE, edgecolor=BLUE, linewidth=1.5)
    axes.text(5.33, 2.91, 'CANONICAL', color=WHITE, fontsize=8.0, fontweight='bold', ha='center')
    axes.text(5.33, 2.55, 'FRAME i', color=WHITE, fontsize=16, fontweight='bold', ha='center')
    axes.text(5.33, 2.29, 'one shared identity', color=WHITE, fontsize=7.5, ha='center')

    add_arrow(axes, (2.71, 3.50), (4.46, 2.91))
    add_arrow(axes, (3.88, 1.80), (4.46, 2.37))


def draw_outputs(axes: Axes) -> None:
    add_box(axes, 6.88, 3.12, 2.58, 0.94, facecolor=WHITE, edgecolor=BLUE)
    axes.text(8.17, 3.74, 'On-screen preview', color=INK, fontsize=10, fontweight='bold', ha='center')
    axes.text(8.17, 3.43, 'pixels + overlay + crop at i', color=MUTED, fontsize=8.2, ha='center')

    add_box(axes, 6.88, 1.36, 2.58, 0.94, facecolor=WHITE, edgecolor=BLUE)
    axes.text(8.17, 1.98, 'Exported frame', color=INK, fontsize=10, fontweight='bold', ha='center')
    axes.text(8.17, 1.67, 'same frame identity, metadata and crop', color=MUTED, fontsize=8.2, ha='center')

    add_arrow(axes, (6.20, 2.75), (6.82, 3.42))
    add_arrow(axes, (6.20, 2.57), (6.82, 2.01))
    axes.text(
        8.17,
        0.98,
        'The preview and export cannot drift onto neighboring frames.',
        color=BLUE,
        fontsize=8.7,
        fontweight='bold',
        ha='center',
    )


def draw_seek_inset(axes: Axes) -> None:
    add_box(axes, 6.82, 4.28, 2.72, 0.64, facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.0)
    axes.text(6.99, 4.72, 'Random access', color=ORANGE, fontsize=7.7, fontweight='bold')
    axes.text(8.20, 4.72, 'keyframe', color=INK, fontsize=7.7, ha='center')
    axes.text(8.77, 4.72, '→', color=ORANGE, fontsize=9, ha='center')
    axes.text(9.16, 4.72, 'frame i', color=INK, fontsize=7.7, ha='center')
    axes.text(8.20, 4.45, 'decode forward to the requested index', color=MUTED, fontsize=7.0, ha='center')


def create_figure() -> Figure:
    figure, axes = plt.subplots(figsize=(11.5, 5.4))
    figure.patch.set_facecolor(BACKGROUND)
    axes.set_facecolor(BACKGROUND)
    axes.set_xlim(0, 10)
    axes.set_ylim(0.70, 5.22)
    axes.axis('off')

    axes.text(
        0.50,
        5.12,
        'One frame index binds the complete rendering pipeline',
        color=INK,
        fontsize=14,
        fontweight='bold',
        va='top',
    )
    axes.text(
        0.50,
        4.83,
        'The displayed image and every frame-dependent model result are selected together.',
        color=MUTED,
        fontsize=9,
        va='top',
    )

    draw_frame_strip(axes)
    draw_metadata(axes)
    draw_contract(axes)
    draw_outputs(axes)
    draw_seek_inset(axes)

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
