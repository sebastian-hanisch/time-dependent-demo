"""Zeitabhängiges Routing: früheste Ankunft, Profilsuche über alle Abfahrtszeiten, Warten und die statische Vergleichsplanung.

Zeitabhängiges Dijkstra (Cooke/Halsey 1966, Dreyfus 1969): wie Dijkstra, aber das Label eines Knotens ist die Ankunftszeit, und eine Kante wird zur Ankunftszeit ihres Startknotens ausgewertet. Es ist genau dann korrekt, wenn die
FIFO-Eigenschaft gilt (Orda/Rom 1990): wer später losfährt, kommt nicht früher an. Ohne FIFO kann sich Warten lohnen; erlaubt man es, gilt die FIFO-Eigenschaft für die "Kante mit Warten" wieder.

Profilsuche: die Ankunftszeit am Ziel als stückweise lineare Funktion der Abfahrtszeit (Label-correcting mit Funktionen: Verknüpfen mit einer Kante und Minimum zweier Funktionen); gilt nur bei FIFO.

Alles ist eigene Umsetzung auf dem CSR-Graphen aus td_graph.py; die Referenzen für die Tests (Brute-Force über alle Routen) stehen unten und sind absichtlich naiv."""

import heapq
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from math import ceil

import numpy as np

from td_graph import DAY, K, STEP, arrival, arrival_waiting, route_arrival

INF = float("inf")
EPS = 1e-9


@dataclass
class TDResult:
    s: int
    depart: float
    arrival: list
    parent_arc: list
    order: list = field(default_factory=list)              # Knoten in der Reihenfolge der Festlegung
    counters: dict = field(default_factory=dict)

    def route(self, g, v):
        """Kanten der frühesten Route vom Start nach v (leer, wenn v der Start oder nicht erreichbar ist)."""
        arcs = []
        src = g.source_of_arcs()
        while v != self.s and self.parent_arc[v] >= 0:
            k = self.parent_arc[v]
            arcs.append(k)
            v = int(src[k])
        return arcs[::-1]


def td_dijkstra(g, s, depart, target=None, wait=False):
    """Früheste Ankunft von s aus bei Abfahrt zur Zeit `depart` (mit `target`: Abbruch, sobald das Ziel feststeht). `wait`: Warten vor jeder Kante erlaubt."""
    n = g.n
    ip, ix, tab, wt = g.ip_l, g.ix_l, g.tab_l, g.wait_l
    arr, parc = [INF] * n, [-1] * n
    arr[s] = float(depart)
    heap = [(float(depart), s)]
    done = [False] * n
    order, relax = [], 0
    while heap:
        a, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        order.append(u)
        if target is not None and u == target:
            break
        for k in range(ip[u], ip[u + 1]):
            relax += 1
            v = ix[k]
            na = arrival_waiting(tab[k], wt[k], a) if wait else arrival(tab[k], a)
            if na < arr[v]:
                arr[v], parc[v] = na, k
                heapq.heappush(heap, (na, v))
    return TDResult(int(s), float(depart), arr, parc, order, {"settled": len(order), "relaxed": relax})


def static_dijkstra(g, s, w, target=None):
    """Dijkstra mit festen Kosten `w` (Feld je Kante): Entfernungen, Vorgängerkante, Zähler. Für die statische Planung (Kosten zu einer festen Uhrzeit oder im Tagesmittel)."""
    n = g.n
    ip, ix = g.ip_l, g.ix_l
    wl = np.asarray(w, dtype=float).tolist()
    dist, parc = [INF] * n, [-1] * n
    dist[s] = 0.0
    heap = [(0.0, s)]
    done = [False] * n
    order, relax = [], 0
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        order.append(u)
        if target is not None and u == target:
            break
        for k in range(ip[u], ip[u + 1]):
            relax += 1
            v = ix[k]
            nd = d + wl[k]
            if nd < dist[v]:
                dist[v], parc[v] = nd, k
                heapq.heappush(heap, (nd, v))
    return TDResult(int(s), 0.0, dist, parc, order, {"settled": len(order), "relaxed": relax})


STATIC_KINDS = ("freeflow", "mean")


def static_costs(g, kind):
    """Feste Kosten je Kante für die statische Planung: "freeflow" = Grundfahrzeit (nachts, freie Fahrt), "mean" = Mittel der Fahrzeit über den Tag (Durchschnitts-Karte)."""
    if kind == "freeflow":
        return g.base.copy()
    if kind == "mean":
        return g.tab[:, :-1].mean(axis=1)
    raise ValueError(kind)


def static_route(g, s, t, kind="mean"):
    """Kantenliste der Route, die mit festen Kosten am kürzesten ist (leer, wenn nicht erreichbar oder s == t); None, wenn t unerreichbar."""
    res = static_dijkstra(g, s, static_costs(g, kind), t)
    if s != t and res.parent_arc[t] < 0:
        return None
    return res.route(g, t)


# --- Stückweise lineare Funktionen für die Profilsuche ------------------------------------------------------------------------------------------------

def _simplify(xs, ys):
    """Doppelte Stützstellen und Punkte auf einer Geraden entfernen."""
    ox, oy = [xs[0]], [ys[0]]
    for i in range(1, len(xs)):
        if xs[i] - ox[-1] < 1e-12:
            oy[-1] = min(oy[-1], ys[i])
            continue
        ox.append(xs[i])
        oy.append(ys[i])
    rx, ry = [ox[0]], [oy[0]]
    for i in range(1, len(ox) - 1):
        s0 = (oy[i] - ry[-1]) / (ox[i] - rx[-1])
        s1 = (oy[i + 1] - oy[i]) / (ox[i + 1] - ox[i])
        if abs(s0 - s1) > 1e-10:
            rx.append(ox[i])
            ry.append(oy[i])
    if len(ox) > 1:
        rx.append(ox[-1])
        ry.append(oy[-1])
    return rx, ry


def _ev(F, x):
    xs, ys = F
    i = bisect_right(xs, x) - 1
    if i >= len(xs) - 1:
        return ys[-1]
    if i < 0:
        return ys[0]
    return ys[i] + (ys[i + 1] - ys[i]) * ((x - xs[i]) / (xs[i + 1] - xs[i]))


def identity():
    return [0.0, DAY], [0.0, DAY]


def link(F, row):
    """Die Ankunft am Ende einer Kante als Funktion der Abfahrt am Anfang der Route: G(d) = h(F(d)) mit h(a) = a + tau(a). Neue Stützstellen dort, wo F eine Stützstelle des Tages überschreitet."""
    xs, ys = F
    ox, oy = [xs[0]], [arrival(row, ys[0])]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y1 > y0 + EPS:
            for j in range(int(y0 // STEP) + 1, min(ceil(y1 / STEP) - 1, K - 1) + 1):
                theta = j * STEP
                if y0 + EPS < theta < y1 - EPS:
                    ox.append(xs[i] + (theta - y0) * (xs[i + 1] - xs[i]) / (y1 - y0))
                    oy.append(theta + row[j])
        ox.append(xs[i + 1])
        oy.append(arrival(row, y1))
    return _simplify(ox, oy)


def pl_min(F, G):
    """Das punktweise Minimum zweier Funktionen und ob G irgendwo echt kleiner ist als F (sonst wird F unverändert zurückgegeben)."""
    pts = sorted(set(F[0]) | set(G[0]))
    fv = [_ev(F, x) for x in pts]
    gv = [_ev(G, x) for x in pts]
    if all(g >= f - 1e-7 for f, g in zip(fv, gv)):
        return F, False
    ox, oy = [pts[0]], [min(fv[0], gv[0])]
    for i in range(len(pts) - 1):
        d0, d1 = fv[i] - gv[i], fv[i + 1] - gv[i + 1]
        if d0 * d1 < 0 and abs(d0) > 1e-12 and abs(d1) > 1e-12:
            x = pts[i] + (pts[i + 1] - pts[i]) * d0 / (d0 - d1)
            ox.append(x)
            oy.append(_ev(F, x))
        ox.append(pts[i + 1])
        oy.append(min(fv[i + 1], gv[i + 1]))
    return _simplify(ox, oy), True


def profile_search(g, s, target, prune=True):
    """Ankunftszeit am Ziel als stückweise lineare Funktion der Abfahrtszeit (0:00 bis 24:00), Label-correcting; nur bei FIFO. Ausgabe: (Stützstellen xs, Ankunft ys) und Zähler.
    prune: ein Label, dessen kleinste Ankunft schon größer ist als die größte Ankunft am Ziel, kann das Ziel nie verbessern (bei FIFO und Fahrzeit >= 0)."""
    if not g.is_fifo():
        raise ValueError("Profilsuche braucht die FIFO-Eigenschaft")
    ip, ix, tab = g.ip_l, g.ix_l, g.tab_l
    F = {int(s): identity()}
    queue, inq = deque([int(s)]), {int(s)}
    counters = {"relaxed": 0, "updates": 0, "points_processed": 0}
    ub = INF
    while queue:
        v = queue.popleft()
        inq.discard(v)
        if v == target:
            continue
        Fv = F[v]
        if prune and min(Fv[1]) >= ub:
            continue
        for k in range(ip[v], ip[v + 1]):
            w = ix[k]
            counters["relaxed"] += 1
            H = link(Fv, tab[k])
            counters["points_processed"] += len(Fv[0])
            if w not in F:
                F[w] = H
                improved = True
            else:
                new, improved = pl_min(F[w], H)
                counters["points_processed"] += len(F[w][0]) + len(H[0])
                if improved:
                    F[w] = new
            if improved:
                counters["updates"] += 1
                if w == target:
                    ub = max(F[w][1])
                if w not in inq:
                    queue.append(w)
                    inq.add(w)
    if target not in F:
        return None, counters
    counters["points"] = len(F[target][0])
    return F[target], counters


# --- Route, Reue, Routenwechsel --------------------------------------------------------------------------------------------------------------------------

def best_route(g, s, t, depart, wait=False):
    """Frühestmögliche Route nach t bei Abfahrt zur Zeit `depart`: (Ankunft, Kantenliste, Ergebnis); Ankunft = inf, wenn t unerreichbar ist."""
    res = td_dijkstra(g, s, depart, t, wait)
    return res.arrival[t], res.route(g, t), res


def regret(g, s, t, depart, arcs):
    """Wie viel später kommt man mit der festen Route `arcs` an als mit der zeitabhängig besten (Minuten, Anteil an der besten Fahrzeit)?"""
    best, _, _ = best_route(g, s, t, depart)
    got = route_arrival(g, arcs, depart)
    return got - best, (got - best) / (best - depart) if best > depart else 0.0


def route_changes(g, s, t, departs):
    """Die beste Route je Abfahrtszeit aus `departs`: Liste von (Abfahrt, Ankunft, Kantentupel)."""
    out = []
    for d in departs:
        a, arcs, _ = best_route(g, s, t, d)
        out.append((float(d), a, tuple(arcs)))
    return out


# --- Referenzen für die Tests (absichtlich naiv) ----------------------------------------------------------------------------------------------------------

def brute_force_arrival(g, s, t, depart, waiting=False, limit=300000):
    """Alle einfachen Routen von s nach t durchprobieren und die früheste Ankunft behalten (nur für kleine Netze). Bei FIFO oder erlaubtem Warten genügen einfache Routen."""
    ip, ix = g.ip_l, g.ix_l
    seen, best, count = {s}, [INF], [0]

    def dfs(v, a):
        count[0] += 1
        assert count[0] < limit, "Netz zu groß für die Brute-Force-Referenz"
        if v == t:
            best[0] = min(best[0], a)
            return
        for k in range(ip[v], ip[v + 1]):
            w = ix[k]
            if w not in seen:
                seen.add(w)
                dfs(w, arrival_waiting(g.tab_l[k], g.wait_l[k], a) if waiting else arrival(g.tab_l[k], a))
                seen.discard(w)
    dfs(s, float(depart))
    return best[0]


def brute_force_no_wait_optimum(g, s, t, depart, limit=300000):
    """Wie oben ohne Warten - auch ohne FIFO: die wirklich früheste Ankunft über alle einfachen Routen (ohne Warten kann eine nicht einfache Route besser sein; hier nur einfache)."""
    return brute_force_arrival(g, s, t, depart, waiting=False, limit=limit)
