"""Die Netze der Demo: kleines Netz (die beste Route wechselt mit der Uhrzeit), Baustelle (Überholen: Warten lohnt), Stadtnetz mit Hauptachsen und Zufallsnetz. Alle Graphen und alle Stauverläufe sind eigene Konstruktionen;
Fahrzeiten in Minuten, Uhrzeiten in Minuten seit Mitternacht."""

from dataclasses import dataclass

import numpy as np

import td_constants as C
from td_graph import K, STEP, Graph, from_arcs


@dataclass(frozen=True)
class Network:
    key: str
    graph: Graph
    title: str
    note: str
    source: int
    target: int
    geometric: bool = True
    arterial: tuple = ()           # Stadtnetz: je Kante (Nummer im CSR) ob Hauptachse
    site_arcs: tuple = ()          # Nummern der Kanten mit Baustellen-Verlauf (in beiden Richtungen)


def rush_curve():
    """Staukurve an den K Stützstellen (Anteil des Höchststaus, 0 bis 1)."""
    return np.array([C.RUSH.get(round(j * STEP / 60.0, 2), 0.0) for j in range(K)])


def congestion_profile(strength, amp):
    """Faktor der Fahrzeit an den Stützstellen: 1 bei freier Fahrt, 1 + Stärke * Anteil * Staukurve im Höchststau."""
    return 1.0 + float(strength) * float(amp) * rush_curve()


def site_profile(base, delay=C.SITE_DELAY, start=8.0, end=8.5):
    """Baustelle auf einer Kante mit Grundfahrzeit `base`: bis `start` (Stunden) kostet sie `delay` Minuten extra, dann sinkt der Zuschlag bis `end` linear auf 0 (die Baustelle endet). Die Fahrzeit sinkt schneller als die Uhr läuft
    (Zuschlag / halbe Stunde > 1 Minute je Minute): wer später losfährt, kommt früher an."""
    return 1.0 + np.array([float(np.interp(j * STEP / 60.0, [start, end], [delay, 0.0])) for j in range(K)]) / float(base)


# --- Kleines Netz -----------------------------------------------------------------------------------------------------------------------------

# (Name, x, y)
SMALL_STOPS = [("Start", 0.0, 3.0), ("Ring", 3.0, 5.6), ("Markt", 3.0, 3.0), ("Dorf", 3.0, 0.4), ("Hafen", 1.5, 1.6), ("Messe", 5.0, 1.4), ("Ziel", 8.0, 3.0)]
# (von, nach, Minuten bei freier Fahrt, Verlauf) - ungerichtet; Verlauf: 0 Stadtautobahn, 1 Innenstadt, 2 Landstraße
SMALL_LEGS = [("Start", "Ring", 3, 0), ("Ring", "Ziel", 7, 0), ("Start", "Markt", 5, 1), ("Markt", "Ziel", 6, 1), ("Start", "Dorf", 7, 2), ("Dorf", "Ziel", 8, 2),
              ("Ring", "Markt", 2, 1), ("Markt", "Dorf", 4, 2), ("Start", "Hafen", 4, 1), ("Hafen", "Markt", 3, 1), ("Dorf", "Messe", 5, 2), ("Messe", "Ziel", 4, 2)]
SMALL_AMPS = (1.6, 0.9, 0.15)


def small_network(strength=C.DEFAULT_STRENGTH):
    names = [s[0] for s in SMALL_STOPS]
    idx = {n: i for i, n in enumerate(names)}
    profiles = [congestion_profile(strength, a) for a in SMALL_AMPS]
    g = from_arcs(len(names), [(idx[a], idx[b], t, p) for a, b, t, p in SMALL_LEGS], profiles, [(s[1], s[2]) for s in SMALL_STOPS], names, directed=False)
    return Network("small", g, "Kleines Netz", "", idx["Start"], idx["Ziel"], True)


# --- Baustelle -----------------------------------------------------------------------------------------------------------------------------------

SITE_STOPS = [("Start", 0.0, 3.0), ("Schnell", 2.0, 4.6), ("Umweg", 2.0, 1.4), ("Kreuzung", 4.0, 3.0), ("Ausweich", 6.0, 0.6), ("Ziel", 8.0, 3.0)]
# (von, nach, Minuten, Verlauf): 3 = Baustelle (45 Minuten Zuschlag, der zwischen 8:00 und 8:30 abklingt)
SITE_LEGS = [("Start", "Schnell", 4, 2), ("Schnell", "Kreuzung", 6, 2), ("Start", "Umweg", 8, 2), ("Umweg", "Kreuzung", 9, 2), ("Kreuzung", "Ziel", 20, 3), ("Kreuzung", "Ausweich", 30, 2), ("Ausweich", "Ziel", 30, 2)]


def site_network(strength=C.DEFAULT_STRENGTH):
    names = [s[0] for s in SITE_STOPS]
    idx = {n: i for i, n in enumerate(names)}
    profiles = [congestion_profile(strength, a) for a in SMALL_AMPS] + [site_profile(20.0)]
    g = from_arcs(len(names), [(idx[a], idx[b], t, p) for a, b, t, p in SITE_LEGS], profiles, [(s[1], s[2]) for s in SITE_STOPS], names, directed=False)
    return Network("site", g, "Baustelle", "", idx["Start"], idx["Ziel"], True, (), tuple(int(k) for k in np.flatnonzero(g.prof >= 3)))


# --- Stadtnetz mit Hauptachsen -------------------------------------------------------------------------------------------------------------------

def build_city(side, strength, sites, seed):
    """Raster mit gestörten Kreuzungen (nur Nachbarn im Raster). Jede vierte Zeile und Spalte ist Hauptachse (60 statt 30 km/h bei freier Fahrt), im Stau trifft es die Hauptachse stärker. `sites` Nebenstraßen bekommen eine Baustelle
    (Verlauf mit Überholen). Start unten links, Ziel oben rechts."""
    rng = np.random.default_rng([int(seed), 808])
    n = side * side
    ij = np.stack(np.divmod(np.arange(n), side), axis=1)
    xy = np.stack([ij[:, 1], ij[:, 0]], axis=1) * C.SPACING + rng.uniform(-C.JITTER, C.JITTER, (n, 2)) * C.SPACING
    arcs, arterial = [], []
    for i in range(side):
        for j in range(side):
            for di, dj in ((0, 1), (1, 0)):
                i2, j2 = i + di, j + dj
                if i2 >= side or j2 >= side:
                    continue
                u, v = i * side + j, i2 * side + j2
                m = float(np.hypot(*(xy[u] - xy[v])))
                art = (i % C.AXIS_EVERY == 0) if di == 0 else (j % C.AXIS_EVERY == 0)
                speed = (C.SPEED_ARTERIAL if art else C.SPEED_LOCAL) * 1000.0 / 60.0          # Meter je Minute
                arcs.append([u, v, m / speed * (1.0 + C.NOISE * rng.random()), 1 if art else 0])
                arterial.append(art)
    local = [i for i, a in enumerate(arterial) if not a]
    chosen = rng.choice(len(local), size=min(int(sites), len(local)), replace=False) if sites and local else []
    profiles = [congestion_profile(strength, C.AMP_LOCAL), congestion_profile(strength, C.AMP_ARTERIAL), congestion_profile(strength, 0.0)]
    for c in chosen:
        arc = arcs[local[int(c)]]
        profiles.append(site_profile(arc[2]))
        arc[3] = len(profiles) - 1
    g = from_arcs(n, arcs, profiles, xy, directed=False)
    key = {(min(a[0], a[1]), max(a[0], a[1])): art for a, art in zip(arcs, arterial)}
    src = g.source_of_arcs()
    flags = tuple(bool(key[(min(int(u), int(v)), max(int(u), int(v)))]) for u, v in zip(src, g.indices))
    return g, flags


def city_network(side, strength, sites, seed):
    g, flags = build_city(int(side), float(strength), int(sites), int(seed))
    note = ("Erzeugtes Stadtnetz: jede vierte Zeile und Spalte ist eine Hauptachse (orange), bei freier Fahrt doppelt so schnell wie die Nebenstraßen, im Stau aber stärker betroffen. Fahrzeiten in Minuten; Start unten links, Ziel oben rechts."
            + (f" {int(sites)} Nebenstraßen haben eine Baustelle (rote Kreuze)." if sites else ""))
    return Network("city", g, "Stadtnetz mit Hauptachsen", note, 0, g.n - 1, True, flags, tuple(int(k) for k in np.flatnonzero(g.prof >= 3)))


# --- Zufallsnetz -----------------------------------------------------------------------------------------------------------------------------------

def build_random(n, degree, strength, seed):
    """Zusammenhängender ungerichteter Zufallsgraph (jeder Knoten hängt an einem früheren, dann kommen zufällige Kanten bis zum mittleren Grad). Grundfahrzeit 2 bis 8 Minuten; 35 % der Kanten sind Schnellstraßen
    (halbe Grundfahrzeit, im Stau stark betroffen), der Rest Nebenstraßen."""
    rng = np.random.default_rng([int(seed), 909])
    pairs = {(int(rng.integers(0, i)), i) for i in range(1, n)}
    target_m = int(round(n * degree / 2))
    while len(pairs) < target_m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v and (min(u, v), max(u, v)) not in pairs:
            pairs.add((min(u, v), max(u, v)))
    arcs = []
    for u, v in sorted(pairs):
        art = rng.random() < 0.35
        b = float(rng.integers(2, 9)) * (0.5 if art else 1.0)
        arcs.append((u, v, b, 1 if art else 0))
    profiles = [congestion_profile(strength, C.AMP_LOCAL), congestion_profile(strength, C.AMP_ARTERIAL)]
    return from_arcs(n, arcs, profiles, rng.random((n, 2)) * 1000.0, directed=False)


def _farthest_by_hops(g, s=0):
    dist = [-1] * g.n
    dist[s] = 0
    q = [s]
    for u in q:
        for v in g.out(u):
            if dist[int(v)] < 0:
                dist[int(v)] = dist[u] + 1
                q.append(int(v))
    return int(max(range(g.n), key=lambda v: (dist[v], -v)))


def random_network(n, degree, strength, seed):
    g = build_random(int(n), float(degree), float(strength), int(seed))
    return Network("random", g, "Zufallsnetz", "Erzeugter Zufallsgraph ohne Raster: 35 % der Kanten sind Schnellstraßen (halbe Grundfahrzeit), die im Stau stärker leiden. Keine Karte, nur Punkte; das Ziel ist der vom Start am weitesten entfernte Knoten (nach Kantenzahl).",
                   0, _farthest_by_hops(g), False)


# --- Zusammenbau -------------------------------------------------------------------------------------------------------------------------------

def make_network(net, strength=C.DEFAULT_STRENGTH, side=C.DEFAULT_SIDE, sites=C.DEFAULT_SITES, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, seed=C.DEFAULT_SEED):
    if net == "small":
        return small_network(strength)
    if net == "site":
        return site_network(strength)
    if net == "city":
        return city_network(side, strength, sites, seed)
    if net == "random":
        return random_network(nodes, degree, strength, seed)
    raise ValueError(net)
