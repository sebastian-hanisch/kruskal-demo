"""Plotly-Abbildungen: Karte mit den Kandidatenkanten, Schrittwiedergabe von Kruskal (angenommene Kanten, verworfene Kanten, aktuelle Kante, Komponenten farbig),
Ergebnis-Karte (MST, Kürzeste-Wege-Baum, Umweg-Faktor je Filiale), Union-Find-Balken und -Verlauf, Sweeps. Achsen sind gesperrt (fixedrange), damit Touch-Geräte
beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MST_COLOR = "#2F6B65"
SPT_COLOR = "#f58518"
REJECT_COLOR = "#e45756"
GREY = "rgba(150,150,150,0.35)"
MODE_COLORS = {"naive": "#e45756", "compress": "#f58518", "rank": "#4c78a8", "full": "#54a24b"}
MODE_LABELS = {"naive": "naiv", "compress": "Pfadhalbierung", "rank": "nach Rang", "full": "Rang + Pfadhalbierung"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.1):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_axes(fig, height):
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _segments(inst, edge_ids):
    xs, ys = [], []
    for i in edge_ids:
        u, v, _w = inst.edges[i]
        xs += [inst.xy[u, 0], inst.xy[v, 0], None]
        ys += [inst.xy[u, 1], inst.xy[v, 1], None]
    return xs, ys


def _line(fig, inst, edge_ids, name, color, width, dash="solid", showlegend=True):
    if len(edge_ids):
        xs, ys = _segments(inst, edge_ids)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend))


def _nodes(fig, inst, colors=None, size=8, name="Filialen", labels=True, showlegend=False):
    n = inst.n
    idx = list(range(1, n)) if inst.kind == "depot" else list(range(n))
    text = None
    if inst.labels is not None:
        text = [inst.labels[i] for i in idx]
    marker = dict(size=size, color=colors if colors is not None else "#4c78a8", line=dict(width=1, color="white"))
    fig.add_trace(go.Scatter(x=inst.xy[idx, 0], y=inst.xy[idx, 1], mode="markers+text" if text else "markers", text=text, textposition="top center", marker=marker,
                             name=name, hoverinfo="skip", showlegend=showlegend))
    if inst.kind == "depot":
        fig.add_trace(go.Scatter(x=[inst.xy[0, 0]], y=[inst.xy[0, 1]], mode="markers", marker=dict(size=15, symbol="star", color="#2ca02c", line=dict(width=1, color="white")),
                                 name="Depot", hoverinfo="skip"))


def build_instance(inst):
    fig = go.Figure()
    _line(fig, inst, range(inst.m), "Kandidatenkanten", GREY, 1, showlegend=True)
    _nodes(fig, inst, size=9)
    return _map_axes(fig, 460 if inst.kind == "depot" else 360)


def _component_colors(inst, accepted):
    parent = list(range(inst.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in accepted:
        u, v, _w = inst.edges[i]
        parent[find(u)] = find(v)
    roots = [find(x) for x in range(inst.n)]
    sizes = {}
    for r in roots:
        sizes[r] = sizes.get(r, 0) + 1
    palette = {}
    colors = []
    for r in roots:
        if sizes[r] == 1:
            colors.append("rgba(150,150,150,0.8)")
        else:
            if r not in palette:
                palette[r] = f"hsl({(len(palette) * 137) % 360},60%,45%)"
            colors.append(palette[r])
    return colors


def build_step_map(inst, steps, k):
    """Zustand nach den ersten `k` betrachteten Kanten: dicke Linie = angenommen, gestrichelt rot = verworfen (schloss einen Kreis), orange = aktuelle Kante; Punktfarbe = Komponente."""
    fig = go.Figure()
    k = max(0, min(k, len(steps)))
    done = steps[:k]
    considered = {i for i, _a, _c in done}
    accepted = [i for i, a, _c in done if a]
    rejected = [i for i, a, _c in done if not a]
    _line(fig, inst, [i for i in range(inst.m) if i not in considered], "noch nicht betrachtet", "rgba(150,150,150,0.22)", 1)
    _line(fig, inst, rejected, "verworfen (Kreis)", REJECT_COLOR, 1.5, "dash")
    _line(fig, inst, accepted, "angenommen", MST_COLOR, 3.5)
    if k >= 1:
        cur = done[-1][0]
        _line(fig, inst, [cur], "aktuelle Kante", "#ff7f0e", 6)
    _nodes(fig, inst, colors=_component_colors(inst, accepted), size=10)
    return _map_axes(fig, 470 if inst.kind == "depot" else 360)


def build_result(inst, tree, spt_tree, stretch_by_node):
    """MST (dick) und, wo er abweicht, der Kürzeste-Wege-Baum (gestrichelt orange); Punktfarbe = Umweg-Faktor der Filiale (Weg im MST / kürzester Weg)."""
    fig = go.Figure()
    in_tree, in_spt = set(tree), set(spt_tree)
    _line(fig, inst, [i for i in range(inst.m) if i not in in_tree and i not in in_spt], "Kandidatenkante", "rgba(150,150,150,0.15)", 1, showlegend=False)
    _line(fig, inst, [i for i in spt_tree if i not in in_tree], "Kürzeste-Wege-Baum (nur die abweichenden Kanten)", SPT_COLOR, 2.5, "dash")
    _line(fig, inst, tree, "billigster Baum (MST)", MST_COLOR, 3.5)
    idx = list(range(1, inst.n)) if inst.kind == "depot" else list(range(inst.n))
    vals = [stretch_by_node[i] for i in idx]
    fig.add_trace(go.Scatter(x=inst.xy[idx, 0], y=inst.xy[idx, 1], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=10, color=vals, colorscale="YlOrRd", cmin=1.0, cmax=max(1.5, max(vals) if vals else 1.5), showscale=True, line=dict(width=1, color="white"),
                                         colorbar=dict(title="Umweg-Faktor", thickness=10, len=0.6))))
    if inst.kind == "depot":
        fig.add_trace(go.Scatter(x=[inst.xy[0, 0]], y=[inst.xy[0, 1]], mode="markers", marker=dict(size=15, symbol="star", color="#2ca02c", line=dict(width=1, color="white")),
                                 name="Depot", hoverinfo="skip"))
    return _map_axes(fig, 480 if inst.kind == "depot" else 360)


def build_uf_bars(table, title_random=None):
    """Zeigerschritte und größte Tiefe je Union-Find-Stufe für EINE Instanz."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Zeigerschritte gesamt", "größte Tiefe im Wald am Ende"), horizontal_spacing=0.12)
    modes = list(table)
    names = [MODE_LABELS[m] for m in modes]
    colors = [MODE_COLORS[m] for m in modes]
    fig.add_trace(go.Bar(x=names, y=[table[m][0] for m in modes], marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=[table[m][2] for m in modes], marker_color=colors, showlegend=False), row=1, col=2)
    return _base(fig, 320)


def build_uf_experiment(rows, kind):
    """Zeigerschritte über n je Stufe, `kind` = 'random' oder 'chain' (logarithmische Achsen)."""
    fig = go.Figure()
    xs = [r["n"] for r in rows]
    for mode, color in MODE_COLORS.items():
        fig.add_trace(go.Scatter(x=xs, y=[max(1.0, r[kind][mode]) for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=MODE_LABELS[mode]))
    fig.update_xaxes(title_text="Knoten n", type="log")
    fig.update_yaxes(title_text="Zeigerschritte (Median)" if kind == "random" else "Zeigerschritte", type="log")
    return _base(fig, 340, legend_y=-0.3)


def build_sweep(rows, param_label, series, y_label, ref_line=None, ref_label=None, log_y=False):
    """`series` = [(key, Name, Farbe)]: Median als Linie, 10. bis 90. Perzentil als Band (`<key>_lo`/`<key>_hi`)."""
    xs = [str(r["value"]) for r in rows]
    fig = go.Figure()
    for key, name, color in series:
        ys = [r[key] for r in rows]
        lo = [r[f"{key}_lo"] for r in rows]
        hi = [r[f"{key}_hi"] for r in rows]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.13)", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=name))
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color="#888", dash="dash", width=1.5), annotation_text=ref_label, annotation_position="top left")
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text=y_label, type="log" if log_y else "linear")
    return _base(fig, 360, legend_y=-0.3)
