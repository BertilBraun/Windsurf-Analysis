from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from server.inference.bot_sort.kalman_filter import KFState
from server.inference.src.common_types import Track


@dataclass
class EdgeRecord:
    i: int
    j: int
    motion_nll: float
    appearance_nll: float
    gap_nll: float
    total: float
    gap_frames: int
    # Optional extras for callbacks; typed as Any upstream
    kf_end: KFState | None = None
    A_ref: Track | None = None
    B_ref: Track | None = None


def show_graph_interactive(
    fragments: List[Track],
    edge_records: List[EdgeRecord],
    on_edge_click: Callable[[EdgeRecord], None] | None = None,
    on_node_click: Callable[[Track], None] | None = None,
) -> None:
    if not edge_records:
        return

    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    G = nx.DiGraph()

    for idx, frag in enumerate(fragments):
        label = f'{idx} (id={frag.track_id})\n[{frag.start_frame}-{frag.end_frame}]'
        G.add_node(idx, label=label, start=frag.start_frame, end=frag.end_frame)

    for rec in edge_records:
        G.add_edge(rec.i, rec.j, total=float(rec.total))

    SRC = 'SOURCE'
    SNK = 'SINK'
    G.add_node(SRC)
    G.add_node(SNK)

    all_nodes = [n for n in G.nodes if isinstance(n, int)]
    indeg = {n: 0 for n in all_nodes}
    outdeg = {n: 0 for n in all_nodes}
    for rec in edge_records:
        outdeg[rec.i] += 1
        indeg[rec.j] += 1
    for n in all_nodes:
        if indeg[n] == 0:
            G.add_edge(SRC, n, total=0.0)
        if outdeg[n] == 0:
            G.add_edge(n, SNK, total=0.0)

    min_start = min(f.start_frame for f in fragments) if fragments else 0
    max_end = max(f.end_frame for f in fragments) if fragments else 1
    span = max(1, max_end - min_start)

    pos: Dict[object, tuple] = {}
    for idx, frag in enumerate(fragments):
        x = (frag.start_frame - min_start) / span
        rnd = float(np.random.RandomState(idx * 9973 + 811).rand())
        y = 0.08 + 0.84 * rnd
        pos[idx] = (x, y)
    pos[SRC] = (-0.08, 0.5)
    pos[SNK] = (1.08, 0.5)

    fig, ax = plt.subplots(figsize=(max(6, len(fragments) * 0.9), 6))
    ax.set_title('ILP Fragment Graph (click node for bbox, edge for comparison)')
    ax.set_axis_off()

    node_labels = {n: (G.nodes[n]['label'] if isinstance(n, int) else n) for n in G.nodes}
    int_nodes = [n for n in G.nodes if isinstance(n, int)]
    nodes_main = nx.draw_networkx_nodes(G, pos, nodelist=int_nodes, node_color='#99c2ff', node_size=800, ax=ax)
    nodes_main.set_picker(True)
    nx.draw_networkx_nodes(G, pos, nodelist=[SRC, SNK], node_color='#dddddd', node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)

    artist_to_record: Dict[object, EdgeRecord] = {}
    for rec in edge_records:
        i = rec.i
        j = rec.j
        p0 = pos[i]
        p1 = pos[j]
        arrow = mpatches.FancyArrowPatch(
            p0, p1, arrowstyle='-|>', mutation_scale=12, color='#444444', linewidth=1.2, alpha=0.9
        )
        arrow.set_picker(True)
        ax.add_patch(arrow)
        artist_to_record[arrow] = rec
        mx = (p0[0] + p1[0]) / 2.0
        my = (p0[1] + p1[1]) / 2.0
        txt = ax.text(
            mx,
            my,
            f'{rec.total:.2f}',
            fontsize=8,
            color='#222222',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7),
        )
        txt.set_picker(True)
        artist_to_record[txt] = rec

    for u, v in G.edges:
        if isinstance(u, str) or isinstance(v, str):
            p0 = pos[u]
            p1 = pos[v]
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    p0, p1, arrowstyle='-|>', mutation_scale=12, color='#bbbbbb', linewidth=1.0, alpha=0.8
                )
            )

    node_index_to_id = {i: node_id for i, node_id in enumerate(int_nodes)}

    def on_pick(event):
        artist = event.artist
        if artist is nodes_main:
            ind_list = getattr(event, 'ind', None)
            if not ind_list:
                return
            pick_idx = int(ind_list[0])
            node_id = node_index_to_id.get(pick_idx)
            if node_id is None:
                return
            frag = fragments[node_id]
            if on_node_click is not None:
                on_node_click(frag)
            return

        rec = artist_to_record.get(artist)
        if not rec:
            return

        if on_edge_click is not None:
            on_edge_click(rec)

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()
