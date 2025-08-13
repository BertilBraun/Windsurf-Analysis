# track_visualization.py
"""Visualize object‑tracking *tracklets* either as a **timeline** (Gantt‑style
chart) *or* as the original **Graphviz graph** of merge candidates.

Usage overview
──────────────
```python
from track_visualization import visualize_tracks

# timeline view (horizontal bars)
fig = visualize_tracks(tracks, style='timeline', gap_threshold=8)
fig.show()

# graph view (nodes + edges)
visualize_tracks(tracks, style='graph', gap_threshold=8)
```

Features
────────
1. **Native dataclass support** – Accepts your `Track` objects (with
   `.start.frame_idx`, `.end.frame_idx`, `.track_id`).
2. **Two visualization styles**
   • *timeline*  – Each track is a horizontal bar aligned to frame indices.
                  Merge candidates appear as dashed arrows between bars.
   • *graph*     – The prior Graphviz node‑edge representation.
3. **Pluggable merge filter** – Pass a lambda ``(src, dst, gap) -> bool`` to keep only
   certain edges (appearance similarity, geographic distance…).
4. **Fully static – no GUI dependencies beyond matplotlib & graphviz.**
"""

from __future__ import annotations

from typing import Dict, List, Literal, Sequence, Tuple

import matplotlib.pyplot as plt


from common_types import *

###############################################################################
# Interval helpers
###############################################################################


def _intervals_from_objects(track_objs: Sequence[Track]) -> Dict[TrackId, Tuple[int, int]]:
    out: Dict[TrackId, Tuple[int, int]] = {}
    for t in track_objs:
        start_f, end_f = t.start.frame_idx, t.end.frame_idx
        if t.track_id in out:
            raise ValueError(f'Duplicate track_id {t.track_id}.')
        out[t.track_id] = (start_f, end_f)
    return out


def track_intervals(tracks: Sequence[Track]) -> Dict[TrackId, Tuple[int, int]]:
    """Return ``{track_id: (start_frame, end_frame)}``."""
    return _intervals_from_objects(tracks)


###############################################################################
# Edge discovery (temporal‑gap heuristic)
###############################################################################


def possible_merges(
    intervals: Dict[TrackId, Tuple[int, int]],
    *,
    gap_threshold: int,
) -> List[Tuple[TrackId, TrackId, int]]:
    """Return ``(src_id, dst_id, gap)`` edges with gap ∈ (0, gap_threshold]."""
    edges: List[Tuple[TrackId, TrackId, int]] = []
    for src_id, (_s_start, s_end) in intervals.items():
        for dst_id, (d_start, _d_end) in intervals.items():
            if src_id == dst_id:
                continue
            gap = d_start - s_end
            if 0 < gap <= gap_threshold:
                edges.append((src_id, dst_id, gap))
    return edges


###############################################################################
# Visualization – Timeline (matplotlib)
###############################################################################


def _visualize_timeline(
    intervals: Dict[TrackId, Tuple[int, int]],
    edges: List[Tuple[TrackId, TrackId, int]],
    title: str,
    *,
    bar_height: float = 0.6,
    figsize: Tuple[int, int] | None = None,
):
    """Return a matplotlib *Figure* with tracks as horizontal bars."""
    n_tracks = len(intervals)
    if figsize is None:
        figsize = (10, max(3, int(n_tracks * 0.6)))

    # Sort tracks by start frame for nicer ordering
    sorted_items = sorted(intervals.items(), key=lambda kv: kv[1][0])
    y_positions = {tid: i for i, (tid, _) in enumerate(sorted_items)}

    fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    for tid, (start, end) in sorted_items:
        ax.broken_barh([(start, end - start)], (y_positions[tid] - bar_height / 2, bar_height), label=f'Track {tid}')

    # Draw merge arrows
    for src, dst, _gap in edges:
        y_src = y_positions[src]
        y_dst = y_positions[dst]
        x_src = intervals[src][1]  # end frame of src
        x_dst = intervals[dst][0]  # start frame of dst
        # Place arrow slightly above bar
        ax.annotate(
            '',
            xy=(x_dst, y_dst),
            xytext=(x_src, y_src),
            arrowprops=dict(arrowstyle='->', linestyle='--', linewidth=1),
            annotation_clip=False,
        )

    # Styling
    ax.set_xlabel('Frame index')
    ax.set_yticks([y_positions[tid] for tid, _ in sorted_items])
    ax.set_yticklabels([f'Track {tid}' for tid, _ in sorted_items])
    ax.invert_yaxis()  # earliest track on top
    ax.grid(True, axis='x', linestyle=':', linewidth=0.5)
    ax.set_title(title)

    return fig


###############################################################################
# Visualization – Graph (Graphviz) [existing]
###############################################################################


def _visualize_graph(
    intervals: Dict[TrackId, Tuple[int, int]],
    edges: List[Tuple[TrackId, TrackId, int]],
    title: str,
    *,
    rankdir: str = 'LR',
    node_shape: str = 'box',
    edge_color: str | None = None,
    outfile_basename: str | None = 'track_graph',
    view: bool = True,
):
    from graphviz import Digraph

    g = Digraph('Tracks', format='png')
    g.attr(rankdir=rankdir)

    for tid, (start, end) in intervals.items():
        g.node(str(tid), label=f'{title} {tid}\n[{start}–{end}]', shape=node_shape)

    edge_attr = {'color': edge_color} if edge_color else {}
    for src, dst, gap in edges:
        g.edge(str(src), str(dst), label=f'gap={gap}', **edge_attr)

    if outfile_basename is not None:
        g.render(outfile_basename, view=view)
    return g


###############################################################################
# Unified public API
###############################################################################


def visualize_tracks(
    tracks: Sequence[Track],
    title: str,
    *,
    style: Literal['timeline', 'graph'] = 'timeline',
    gap_threshold: int = 30,
    **style_kwargs,
):
    """Visualize *tracks* either as a timeline (matplotlib) or a Graphviz graph.

    Parameters
    ----------
    tracks : Sequence[Track]
        Your list of Track dataclass objects.
    style : {"timeline", "graph"}
        Select visualization backend.
    gap_threshold : int
        Max gap (frames) for a merge edge *unless* ``merge_filter`` provided.
    merge_filter : callable, optional
        Custom predicate ``(src_id, dst_id, gap) -> bool``. Replaces
        ``gap_threshold``.
    **style_kwargs
        Passed straight to the chosen backend helper:
            • timeline → ``_visualize_timeline`` (e.g., figsize=(12,4))
            • graph    → ``_visualize_graph`` (e.g., rankdir="TB", outfile_basename=None)
    """
    intervals = track_intervals(tracks)

    edges = possible_merges(intervals, gap_threshold=gap_threshold)

    if style == 'timeline':
        _visualize_timeline(intervals, edges, title, **style_kwargs)
        # show the figure and wait for the user to close it
        plt.show()
        plt.close()
    elif style == 'graph':
        _visualize_graph(intervals, edges, title, **style_kwargs)
    else:
        raise ValueError("style must be 'timeline' or 'graph'")
