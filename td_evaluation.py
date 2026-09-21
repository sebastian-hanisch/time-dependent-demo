"""Kennzahlen, Tageslauf der besten Route, Bildfolge und die Messreihen der Experimente. Aufwand als Zähler (festgelegte Knoten, Kantenprüfungen, Stützstellen); Laufzeiten stehen nur als Messwerte in der App."""

import time
from dataclasses import dataclass, field

import numpy as np

import td_algorithm as alg
import td_constants as C
from td_graph import DAY, route_arcs_nodes, route_arrival
from td_scenario import make_network

DAY_STEP = 15                                    # Minuten zwischen zwei Abfahrtszeiten des Tageslaufs
DEPARTS = tuple(range(0, int(DAY), DAY_STEP))    # 0:00 bis 23:45
CURVE_STEP = 10                                  # Abstand der Abfahrtszeiten der Kurve, wenn die Profilsuche nicht möglich ist (ohne FIFO)


@dataclass
class Analysis:
    net: object
    fifo: bool
    static_arcs: dict                            # Art der statischen Planung -> Kantenliste (oder None)
    day: list                                    # (Abfahrt, Ankunft, Kanten) der besten Route je Abfahrtszeit des Tageslaufs
    segments: list                               # aufeinanderfolgende Abfahrten mit derselben Route: (von, bis, Kanten)
    curve_x: list                                # Abfahrtszeiten der Kurve (Minuten)
    curve: dict                                  # Name -> Fahrzeit in Minuten je Abfahrtszeit
    profile_points: int
    profile_counters: dict
    seconds: dict = field(default_factory=dict)


def _segments(day):
    out = []
    for d, _, arcs in day:
        if out and out[-1][2] == arcs:
            out[-1][1] = d
        else:
            out.append([d, d, arcs])
    return [(a, b + DAY_STEP, arcs) for a, b, arcs in out]


def analyse(net):
    g, s, t = net.graph, net.source, net.target
    fifo = g.is_fifo()
    t0 = time.perf_counter()
    day = alg.route_changes(g, s, t, DEPARTS)
    t_day = time.perf_counter() - t0
    static = {kind: alg.static_route(g, s, t, kind) for kind in alg.STATIC_KINDS}
    reachable = all(np.isfinite(a) for _, a, _ in day)
    curve, points, counters, t_prof = {}, 0, {}, 0.0
    if not reachable:
        return Analysis(net, fifo, static, day, [], [], {}, 0, {}, {"day": t_day})
    if fifo:
        t0 = time.perf_counter()
        F, counters = alg.profile_search(g, s, t)
        t_prof = time.perf_counter() - t0
        points = len(F[0])
        xs = list(np.arange(0.0, DAY, CURVE_STEP)) + [DAY]
        curve["best"] = [alg._ev(F, x) - x for x in xs]
    else:
        xs = list(np.arange(0.0, DAY, CURVE_STEP)) + [DAY]
        curve["best"] = [alg.best_route(g, s, t, x)[0] - x for x in xs]
        curve["wait"] = [alg.best_route(g, s, t, x, wait=True)[0] - x for x in xs]
    for kind, arcs in static.items():
        if arcs is not None:
            curve[kind] = [route_arrival(g, arcs, x) - x for x in xs]
    return Analysis(net, fifo, static, day, _segments(day), xs, curve, points, counters, {"day": t_day, "profile": t_prof})


def at(a, depart):
    """Kennzahlen bei Abfahrt zur Zeit `depart` (Minuten seit Mitternacht)."""
    net = a.net
    g, s, t = net.graph, net.source, net.target
    arr, arcs, res = alg.best_route(g, s, t, depart)
    if not np.isfinite(arr):
        return {"reachable": False}
    static_res = alg.static_dijkstra(g, s, alg.static_costs(g, "mean"), t)
    st_arcs = a.static_arcs["mean"]
    st_arr = route_arrival(g, st_arcs, depart)
    m = {"reachable": True, "time": arr - depart, "arrival": arr, "arcs": arcs, "static_time": st_arr - depart, "static_arcs": st_arcs, "regret": st_arr - arr,
         "regret_rel": (st_arr - arr) / (arr - depart) if arr > depart else 0.0, "same_route": tuple(st_arcs) == tuple(arcs),
         "settled": res.counters["settled"], "relaxed": res.counters["relaxed"], "static_settled": static_res.counters["settled"], "static_relaxed": static_res.counters["relaxed"],
         "free_time": alg.route_arrival(g, arcs, 0.0) - 0.0 if False else None}
    ff = a.static_arcs["freeflow"]
    m["freeflow_time"] = route_arrival(g, ff, depart) - depart
    m["routes_over_day"] = len({arcs for _, _, arcs in a.day})
    if not a.fifo:
        w_arr, w_arcs, _ = alg.best_route(g, s, t, depart, wait=True)
        m.update(wait_time=w_arr - depart, wait_arcs=w_arcs, wait_gain=arr - w_arr)
    return m


def frames(g, order, max_frames=60):
    """Welche Anzahlen festgelegter Knoten gezeigt werden (0 bis Ende, höchstens `max_frames` + 1 Bilder)."""
    n = len(order)
    if n <= max_frames:
        return list(range(n + 1))
    return sorted({round(i * n / max_frames) for i in range(max_frames + 1)} | {0, n})


# --- Messreihen ---------------------------------------------------------------------------------------------------------------------------------------------

def _mean(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def pick_pairs(net, k, seed):
    """k Start-Ziel-Paare mit mindestens `H` Kanten Abstand (Stadtnetz: die Seitenlänge, sonst 4), damit es überhaupt Alternativen gibt."""
    g = net.graph
    rng = np.random.default_rng([int(seed), 55])
    need = int(round(np.sqrt(g.n))) if net.key == "city" else 4
    out = []
    tries = 0
    while len(out) < k and tries < 5000:
        tries += 1
        s = int(rng.integers(0, g.n))
        hop = {s: 0}
        q = [s]
        for u in q:
            for v in g.out(u):
                if int(v) not in hop:
                    hop[int(v)] = hop[u] + 1
                    q.append(int(v))
        far = [v for v, h in hop.items() if h >= need]
        if far:
            out.append((s, int(rng.choice(far))))
    return out


def regret_by_strength(net_key="city", strengths=(0.5, 1.0, 1.5, 2.0, 3.0), departs=(180, 480, 720, 1020), seeds=C.SWEEP_SEEDS, pairs=20, side=8, nodes=C.DEFAULT_NODES):
    """Wie viel später (in Prozent der besten Fahrzeit) kommt man an, wenn man mit den Tagesmittel-Kosten plant und dann zur Uhrzeit fährt? Dazu der Anteil der Paare, deren beste Route um 8:00 eine andere ist als um 3:00."""
    rows = []
    for st in strengths:
        acc = []
        for sd in seeds:
            net = make_network(net_key, strength=st, side=side, nodes=nodes, seed=sd)
            g = net.graph
            for s, t in pick_pairs(net, pairs, sd):
                plan = alg.static_route(g, s, t, "mean")
                night, peak = alg.best_route(g, s, t, 180)[1], alg.best_route(g, s, t, 480)[1]
                row = {"changes": float(night != peak)}
                for d in departs:
                    row[f"regret_{d}"] = alg.regret(g, s, t, d, plan)[1]
                acc.append(row)
        rows.append({"strength": st, **_mean(acc)})
    return rows


def regret_worst(net_key="city", strength=1.5, depart=480, seeds=C.SWEEP_SEEDS, pairs=20, side=8, kind="mean"):
    """Mittel und Höchstwert der Reue (Anteil) zu einer Abfahrtszeit, für die statische Planung `kind` (Tagesmittel oder freie Fahrt)."""
    vals = []
    for sd in seeds:
        net = make_network(net_key, strength=strength, side=side, seed=sd)
        g = net.graph
        for s, t in pick_pairs(net, pairs, sd):
            vals.append(alg.regret(g, s, t, depart, alg.static_route(g, s, t, kind))[1])
    return {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "share_worse": float(np.mean(np.array(vals) > 1e-9))}


def cost_rows(sides=(6, 8, 10, 12), strength=C.DEFAULT_STRENGTH, seeds=C.SWEEP_SEEDS[:3]):
    """Was kostet die Zeitabhängigkeit? Festgelegte Knoten und Kantenprüfungen: zeitabhängiges Dijkstra gegen statisches Dijkstra; Stützstellen und verarbeitete Stützstellen der Profilsuche (Stadtnetz, mittleres Paar)."""
    rows = []
    for side in sides:
        acc = []
        for sd in seeds:
            net = make_network("city", strength=strength, side=side, seed=sd)
            g = net.graph
            s, t = pick_pairs(net, 1, sd)[0]
            td = alg.td_dijkstra(g, s, 480).counters
            st = alg.static_dijkstra(g, s, alg.static_costs(g, "mean")).counters
            _, cnt = alg.profile_search(g, s, t)
            acc.append({"n": g.n, "td_relaxed": td["relaxed"], "static_relaxed": st["relaxed"], "td_settled": td["settled"], "static_settled": st["settled"], "profile_points": cnt["points"], "profile_work": cnt["points_processed"]})
        rows.append({"side": side, **_mean(acc)})
    return rows


def site_rows(sites=(0, 5, 10, 20, 40), side=8, strength=C.DEFAULT_STRENGTH, departs=(420, 450, 480, 500), seeds=C.SWEEP_SEEDS, pairs=15):
    """Wann irrt Dijkstra? Anteil der Abfragen (Paar und Abfahrtszeit), bei denen die Ankunft ohne Warten später ist als mit erlaubtem Warten, und um wie viele Minuten (Mittel über die betroffenen Abfragen)."""
    rows = []
    for k in sites:
        n = worse = 0
        gains = []
        for sd in seeds:
            net = make_network("city", strength=strength, side=side, sites=k, seed=sd)
            g = net.graph
            for s, t in pick_pairs(net, pairs, sd):
                for d in departs:
                    a0 = alg.best_route(g, s, t, d)[0]
                    a1 = alg.best_route(g, s, t, d, wait=True)[0]
                    n += 1
                    if a0 > a1 + 1e-6:
                        worse += 1
                        gains.append(a0 - a1)
        rows.append({"sites": k, "share": worse / n, "gain": float(np.mean(gains)) if gains else 0.0, "max_gain": float(np.max(gains)) if gains else 0.0})
    return rows
