"""Plotly-Abbildungen: Karte mit Ankunftszeiten und Routen, Fahrzeit gegen Abfahrtszeit, Tageslauf der besten Route, Experimente. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import td_constants as C
from td_graph import tau

NAMES = {"best": "beste Route (zeitabhängig)", "mean": "geplant mit Tagesmittel-Kosten", "freeflow": "geplant mit freier Fahrt", "wait": "beste Route mit Warten"}
LINE = {"best": dict(color=C.COLORS["td"], width=3), "mean": dict(color=C.COLORS["static"], width=2), "freeflow": dict(color="#888888", width=2, dash="dot"), "wait": dict(color=C.COLORS["wait"], width=3, dash="dash")}


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _segments(g, mask=None):
    """Kanten als eine Linienspur (None trennt die Segmente); bei ungerichteten Netzen Hin- und Rückrichtung nur einmal."""
    src = g.source_of_arcs()
    dst = g.indices
    keep = np.ones(len(src), dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    if not g.directed:
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        first = np.zeros(len(src), dtype=bool)
        _, idx = np.unique(lo * g.n + hi, return_index=True)
        first[idx] = True
        keep &= first
    u, v = src[keep], dst[keep]
    x = np.full(3 * len(u), None, dtype=object)
    y = np.full(3 * len(u), None, dtype=object)
    x[0::3], x[1::3] = g.xy[u, 0], g.xy[v, 0]
    y[0::3], y[1::3] = g.xy[u, 1], g.xy[v, 1]
    return x, y


def _route_xy(g, s, arcs):
    nodes = [int(s)] + [int(g.indices[k]) for k in arcs]
    return g.xy[nodes]


def build_network(net, res, k, depart, routes=(), height=480):
    """Die Karte, nachdem das zeitabhängige Dijkstra `k` Knoten festgelegt hat (Färbung: Minuten seit der Abfahrt); `routes` = [(Kanten, Name, Farbe, Stil)]. Bei kleinen Netzen mit Namen und der Fahrzeit jeder Kante zur Abfahrtszeit."""
    g = net.graph
    small = bool(g.names)
    fig = go.Figure()
    if net.geometric:
        ex, ey = _segments(g)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.45)", width=1), hoverinfo="skip", showlegend=False))
        if net.arterial and any(net.arterial):
            ax_, ay_ = _segments(g, net.arterial)
            fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", line=dict(color="rgba(230,140,0,0.75)", width=3), name="Hauptachse", hoverinfo="skip"))
    if net.site_arcs:
        mask = np.zeros(g.m, dtype=bool)
        mask[list(net.site_arcs)] = True
        src = g.source_of_arcs()
        mid = (g.xy[src[mask]] + g.xy[g.indices[mask]]) / 2
        fig.add_trace(go.Scatter(x=mid[:, 0], y=mid[:, 1], mode="markers", name="Baustelle", hoverinfo="skip", marker=dict(symbol="x", size=11, color="#d62728", line=dict(width=2))))
    if small:
        src = g.source_of_arcs()
        seen = set()
        for k_, (u, v) in enumerate(zip(src.tolist(), g.indices.tolist())):
            if (v, u) in seen:
                continue
            seen.add((u, v))
            mid = (g.xy[u] + g.xy[v]) / 2
            fig.add_annotation(x=mid[0], y=mid[1], text=f"{tau(g.tab_l[k_], depart):.0f} min", showarrow=False, font=dict(size=10, color="#555"), bgcolor="rgba(255,255,255,0.85)")
    n = g.n
    size = 16 if small else (5 if n > 300 else 8)
    done = res.order[:k]
    arr = np.array([res.arrival[v] - depart for v in done]) if done else np.array([])
    rest = np.setdiff1d(np.arange(n), np.array(done, dtype=int))
    if small:
        text = [f"{g.names[v]}<br>{res.arrival[v] - depart:.0f} min" if v in set(done) else g.names[v] for v in range(n)]
        col = np.full(n, np.nan)
        col[done] = arr
        fig.add_trace(go.Scatter(x=g.xy[:, 0], y=g.xy[:, 1], mode="markers+text", showlegend=False, text=text, textposition="top center", hoverinfo="skip",
                                 marker=dict(size=size, color=col if done else "white", colorscale="Viridis", cmin=0, cmax=max(float(np.nanmax(col)) if done else 1.0, 1.0), showscale=False, line=dict(color="gray", width=1.5))))
    else:
        if len(rest):
            fig.add_trace(go.Scatter(x=g.xy[rest, 0], y=g.xy[rest, 1], mode="markers", showlegend=False, hoverinfo="skip", marker=dict(size=size, color="rgba(200,200,200,0.9)")))
        if done:
            fig.add_trace(go.Scatter(x=g.xy[done, 0], y=g.xy[done, 1], mode="markers", showlegend=False, customdata=arr, hovertemplate="%{customdata:.0f} min nach der Abfahrt<extra></extra>",
                                     marker=dict(size=size, color=arr, colorscale="Viridis", cmin=0, cmax=max(float(arr.max()), 1.0), colorbar=dict(title="Minuten<br>seit Abfahrt", thickness=12, len=0.6))))
    for arcs, name, color, dash in routes:
        pts = _route_xy(g, net.source, arcs)
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=color, width=6 if dash == "solid" else 3, dash=dash), name=name, hoverinfo="skip"))
    for node, name, color, symbol in ((net.source, "Start", C.COLORS["start"], "diamond"), (net.target, "Ziel", C.COLORS["goal"], "star")):
        lab = g.names[node] if small else f"{node}"
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", name=f"{name}: {lab}", hoverinfo="skip", marker=dict(size=15, color=color, symbol=symbol, line=dict(color="white", width=1.5))))
    fig.update_xaxes(visible=False)
    if net.key == "city":
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    if small:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        pad = 0.16 * (hi - lo)
        fig.update_xaxes(range=[lo[0] - pad[0], hi[0] + 1.2 * pad[0]])
        fig.update_yaxes(range=[lo[1] - pad[1], hi[1] + 1.8 * pad[1]])
    return _base(fig, height)


def build_curve(a, depart, height=420):
    """Fahrzeit gegen Abfahrtszeit: die beste Route (bei FIFO exakt aus der Profilsuche), die statisch geplanten Routen zur jeweiligen Uhrzeit gefahren, bei fehlender FIFO-Eigenschaft die beste Route mit Warten."""
    fig = go.Figure()
    x = [v / 60.0 for v in a.curve_x]
    for name in ("freeflow", "mean", "best", "wait"):
        if name in a.curve:
            fig.add_trace(go.Scatter(x=x, y=a.curve[name], mode="lines", name=NAMES[name], line=LINE[name], hovertemplate="%{x:.2f} h: %{y:.1f} min<extra></extra>"))
    fig.add_vline(x=depart / 60.0, line=dict(color="#111111", dash="dash", width=1))
    fig.update_layout(xaxis_title="Abfahrtszeit [h]", yaxis_title="Fahrzeit [min]")
    fig.update_xaxes(range=[0, 24], dtick=3)
    return _base(fig, height)


def route_table(a):
    """Tageslauf der besten Route für kleine Netze: (von, bis, Ortsfolge) je Abschnitt."""
    g = a.net.graph
    rows = []
    for lo, hi, arcs in a.segments:
        nodes = [a.net.source] + [int(g.indices[k]) for k in arcs]
        rows.append({"Abfahrt": f"{C.hhmm(lo)}–{C.hhmm(hi) if hi < 1440 else '24:00'}", "Beste Route": " → ".join(g.names[v] for v in nodes)})
    return rows


def build_regret(rows):
    """Reue der statischen Planung gegen die Stau-Stärke, je Abfahrtszeit (Anteil der besten Fahrzeit, in %)."""
    fig = go.Figure()
    x = [r["strength"] for r in rows]
    for d, name, dash in ((480, "8:00", "solid"), (1020, "17:00", "dash"), (180, "3:00", "dot"), (720, "12:00", "dashdot")):
        fig.add_trace(go.Scatter(x=x, y=[100 * r[f"regret_{d}"] for r in rows], mode="lines+markers", name=f"Abfahrt {name}", line=dict(dash=dash)))
    fig.update_layout(xaxis_title="Stau-Stärke", yaxis_title="Mehrzeit der statisch geplanten Route [%]")
    return _base(fig, 320)


def build_cost(rows):
    """Aufwand: Kantenprüfungen des zeitabhängigen Dijkstra, 96 solche Läufe (alle 15 Minuten) und verarbeitete Stützstellen der Profilsuche, gegen die Knotenzahl."""
    fig = go.Figure()
    x = [r["n"] for r in rows]
    fig.add_trace(go.Scatter(x=x, y=[r["td_relaxed"] for r in rows], mode="lines+markers", name="ein Lauf (zeitabhängig = statisch)"))
    fig.add_trace(go.Scatter(x=x, y=[96 * r["td_relaxed"] for r in rows], mode="lines+markers", name="96 Läufe (alle 15 min)"))
    fig.add_trace(go.Scatter(x=x, y=[r["profile_work"] for r in rows], mode="lines+markers", name="Profilsuche (ganzer Tag)"))
    fig.update_layout(xaxis_title="Knoten", yaxis_title="Kantenprüfungen bzw. verarbeitete Stützstellen")
    fig.update_yaxes(type="log")
    return _base(fig, 320)


def build_sites(rows):
    """Anteil der Abfragen, bei denen Dijkstra ohne Warten später ankommt als mit Warten, gegen die Zahl der Baustellen."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(r["sites"]) for r in rows], y=[100 * r["share"] for r in rows], marker_color=C.COLORS["wait"], name="Abfragen, bei denen Warten hilft"))
    fig.update_layout(xaxis_title="Baustellen im 8 × 8-Stadtnetz", yaxis_title="Anteil der Abfragen [%]")
    return _base(fig, 320)
