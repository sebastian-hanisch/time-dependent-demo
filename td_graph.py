"""Gerichteter Graph in CSR-Form mit zeitabhängigen Fahrzeiten.

Jede Kante k hat eine Grundfahrzeit `base[k]` (Minuten bei freier Fahrt) und eine Verlaufsklasse `prof[k]`; die Fahrzeit zur Abfahrtszeit t (Minuten seit Mitternacht) ist
    tau_k(t) = base[k] * Faktor_(prof[k])(t),
der Faktor ist stückweise linear zwischen Stützstellen alle 30 Minuten (0:00 bis 24:00) und danach konstant. Die Ankunft ist t + tau_k(t).
Nachbarn eines Knotens u stehen in indices[indptr[u]:indptr[u+1]]; Parallelkanten sind erlaubt (jede Kante hat eine eigene Nummer k = Position in indices)."""

from dataclasses import dataclass, field

import numpy as np

STEP = 30.0                        # Minuten zwischen zwei Stützstellen
DAY = 1440.0                       # Minuten eines Tages
K = int(DAY / STEP) + 1            # 49 Stützstellen: 0:00, 0:30, ..., 24:00


@dataclass(frozen=True)
class Graph:
    n: int
    indptr: np.ndarray            # (n + 1,) int
    indices: np.ndarray           # (m,) int   Zielknoten je gerichteter Kante
    base: np.ndarray              # (m,) float Grundfahrzeit in Minuten
    prof: np.ndarray              # (m,) int   Verlaufsklasse der Kante
    profiles: np.ndarray          # (P, K) float Faktor je Klasse an den Stützstellen
    xy: np.ndarray                # (n, 2) Lage für die Zeichnung
    names: tuple = ()
    directed: bool = False
    tab: np.ndarray = field(init=False, repr=False, compare=False)          # (m, K) Fahrzeit an den Stützstellen
    tab_l: list = field(init=False, repr=False, compare=False)
    wait_l: list = field(init=False, repr=False, compare=False)             # (m, K) kleinste Ankunft bei Abfahrt an oder nach der Stützstelle (für das Warten)
    ip_l: list = field(init=False, repr=False, compare=False)
    ix_l: list = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        tab = self.base[:, None] * self.profiles[self.prof] if self.m else np.zeros((0, K))
        arr = tab + np.arange(K) * STEP
        wait = np.minimum.accumulate(arr[:, ::-1], axis=1)[:, ::-1] if self.m else arr
        object.__setattr__(self, "tab", tab)
        object.__setattr__(self, "tab_l", tab.tolist())
        object.__setattr__(self, "wait_l", wait.tolist())
        object.__setattr__(self, "ip_l", self.indptr.tolist())
        object.__setattr__(self, "ix_l", self.indices.tolist())

    @property
    def m(self):
        return len(self.indices)

    def out(self, u):
        return self.indices[self.indptr[u]:self.indptr[u + 1]]

    def degree(self):
        return np.diff(self.indptr)

    def source_of_arcs(self):
        return np.repeat(np.arange(self.n), self.degree())

    def is_fifo(self):
        """Wahr, wenn keine Kante überholt werden kann: die Ankunft t + tau(t) fällt nirgends (die Fahrzeit sinkt nie schneller als 1 Minute je Minute)."""
        return bool(self.fifo_arcs().all())

    def fifo_arcs(self):
        if not self.m:
            return np.zeros(0, dtype=bool)
        return (np.diff(self.tab, axis=1) / STEP >= -1.0 - 1e-9).all(axis=1)


def tau(row, t):
    """Fahrzeit einer Kante (Zeile der Tabelle `tab_l`) bei Abfahrt zur Zeit t: linear zwischen den Stützstellen, davor und danach konstant."""
    if t <= 0.0:
        return row[0]
    if t >= DAY:
        return row[-1]
    j = int(t // STEP)
    return row[j] + (row[j + 1] - row[j]) * ((t - j * STEP) / STEP)


def arrival(row, t):
    return t + tau(row, t)


def arrival_waiting(row, wait_row, t):
    """Früheste Ankunft, wenn man vor der Kante beliebig lange warten darf: das Minimum von t + tau(t') über alle Abfahrten t' >= t (liegt bei t oder einer Stützstelle)."""
    a = t + tau(row, t)
    if t >= DAY:
        return a
    j = int(t // STEP) + 1
    return min(a, wait_row[j])


def from_arcs(n, arcs, profiles, xy, names=(), directed=False):
    """Baut den Graphen aus (u, v, Grundfahrzeit, Klasse)-Tupeln. Ungerichtet: jede Kante einmal angeben, die Rückrichtung entsteht hier. Selbstschleifen fallen weg, Parallelkanten bleiben."""
    rows = []
    for u, v, b, p in arcs:
        u, v, b, p = int(u), int(v), float(b), int(p)
        if u == v:
            continue
        rows.append((u, v, b, p))
        if not directed:
            rows.append((v, u, b, p))
    rows.sort()
    src = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    dst = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
    base = np.fromiter((r[2] for r in rows), dtype=float, count=len(rows))
    prof = np.fromiter((r[3] for r in rows), dtype=np.int64, count=len(rows))
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return Graph(n, indptr, dst, base, prof, np.asarray(profiles, dtype=float), np.asarray(xy, dtype=float), tuple(names), directed)


def route_arrival(g, arcs, depart, waiting=False):
    """Ankunftszeit, wenn man zur Zeit `depart` startet und die Kanten der Liste nacheinander fährt (ohne Warten, oder mit Warten vor jeder Kante)."""
    t = float(depart)
    for k in arcs:
        t = arrival_waiting(g.tab_l[k], g.wait_l[k], t) if waiting else arrival(g.tab_l[k], t)
    return t


def route_arcs_nodes(g, s, arcs):
    """Die Knotenfolge einer Kantenliste (vom Start s aus)."""
    out = [int(s)]
    for k in arcs:
        out.append(int(g.indices[k]))
    return out
