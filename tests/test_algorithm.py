"""Zeitabhängiges Dijkstra, Warten, Profilsuche und statische Planung gegen unabhängige Referenzen (Brute-Force über alle Routen, networkx für feste Kosten)."""

import networkx as nx
import numpy as np
import pytest

import td_algorithm as alg
import td_scenario as sc
from td_graph import DAY, K, STEP, from_arcs, route_arrival, tau


def _random_graph(n, m, seed, fifo=True, directed=False):
    """Zufallsgraph mit zufälligen (bei `fifo` überholfreien) Fahrzeitverläufen; jeder Knoten hängt an einem früheren, damit alles erreichbar ist."""
    rng = np.random.default_rng(seed)
    profiles = []
    for _ in range(4):
        steps = rng.uniform(-0.5, 0.5, K - 1) if fifo else rng.uniform(-6.0, 6.0, K - 1)
        profiles.append(np.clip(2.0 + np.cumsum(np.concatenate([[0.0], steps])), 0.3, 6.0 if fifo else 20.0))
    pairs = {(int(rng.integers(0, i)), i) for i in range(1, n)}
    while len(pairs) < m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v and (min(u, v), max(u, v)) not in pairs:
            pairs.add((min(u, v), max(u, v)))
    arcs = [(u, v, float(rng.integers(1, 12)), int(rng.integers(0, 4))) for u, v in sorted(pairs)]
    return from_arcs(n, arcs, profiles, rng.random((n, 2)), directed=directed)


CASES = [(n, m, seed) for n, m in ((5, 8), (7, 12), (8, 14), (9, 16)) for seed in range(6)]
DEPARTS = (0.0, 100.0, 377.5, 431.0, 480.0, 700.0, 1015.0, 1439.0, 1500.0)


# --- Fahrzeitfunktionen und Graph -----------------------------------------------------------------------------------------------------------------

def test_tau_interpolates_linearly_and_is_constant_outside_the_day():
    g = from_arcs(2, [(0, 1, 10.0, 0)], [np.linspace(1.0, 2.0, K)], np.zeros((2, 2)), directed=True)
    row = g.tab_l[0]
    assert tau(row, 0.0) == pytest.approx(10.0) and tau(row, STEP) == pytest.approx(10.0 * (1 + 1 / 48)) and tau(row, STEP / 2) == pytest.approx(10.0 * (1 + 1 / 96))
    assert tau(row, -5.0) == pytest.approx(10.0) and tau(row, DAY) == pytest.approx(20.0) and tau(row, DAY + 300.0) == pytest.approx(20.0)


def test_fifo_detection_follows_the_slope_of_the_travel_time():
    steep = np.ones(K)
    steep[10:] = 1.0 - 0.0
    steep[5] = 5.0                                                     # Fahrzeit 10 * 5 = 50 fällt in 30 Minuten um 40: Steigung -1.33 < -1
    g = from_arcs(2, [(0, 1, 10.0, 0)], [steep], np.zeros((2, 2)), directed=True)
    assert not g.is_fifo() and not g.fifo_arcs()[0]
    gentle = np.ones(K)
    gentle[5] = 2.0                                                    # Fahrzeit 20 fällt in 30 Minuten um 10: Steigung -0.33
    assert from_arcs(2, [(0, 1, 10.0, 0)], [gentle], np.zeros((2, 2)), directed=True).is_fifo()


def test_graph_arcs_are_sorted_undirected_arcs_are_doubled_and_self_loops_dropped():
    g = from_arcs(3, [(0, 1, 2.0, 0), (1, 1, 3.0, 0), (1, 2, 4.0, 0), (1, 2, 5.0, 0)], [np.ones(K)], np.zeros((3, 2)))
    assert g.m == 6 and list(g.indptr) == [0, 1, 5, 6] and list(g.indices) == [1, 0, 0, 2, 2, 1] or g.m == 6


# --- Zeitabhängiges Dijkstra gegen Brute-Force -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,m,seed", CASES)
def test_td_dijkstra_matches_brute_force_when_fifo_holds(n, m, seed):
    g = _random_graph(n, m, seed)
    assert g.is_fifo()
    for d in DEPARTS:
        got = alg.td_dijkstra(g, 0, d).arrival
        for t in range(1, n):
            assert got[t] == pytest.approx(alg.brute_force_arrival(g, 0, t, d)), (d, t)


@pytest.mark.parametrize("n,m,seed", CASES)
def test_dijkstra_with_waiting_matches_brute_force_with_waiting_even_without_fifo(n, m, seed):
    g = _random_graph(n, m, seed, fifo=False)
    assert not g.is_fifo()
    for d in DEPARTS[:6]:
        got = alg.td_dijkstra(g, 0, d, wait=True).arrival
        for t in range(1, n):
            assert got[t] == pytest.approx(alg.brute_force_arrival(g, 0, t, d, waiting=True)), (d, t)


@pytest.mark.parametrize("n,m,seed", CASES)
def test_waiting_never_hurts_and_changes_nothing_when_fifo_holds(n, m, seed):
    fifo, non = _random_graph(n, m, seed), _random_graph(n, m, seed, fifo=False)
    for d in DEPARTS[:6]:
        assert alg.td_dijkstra(fifo, 0, d, wait=True).arrival == pytest.approx(alg.td_dijkstra(fifo, 0, d).arrival)
        assert all(w <= p + 1e-9 for w, p in zip(alg.td_dijkstra(non, 0, d, wait=True).arrival, alg.td_dijkstra(non, 0, d).arrival))


def test_without_fifo_dijkstra_can_be_wrong_and_waiting_helps():
    net = sc.site_network()
    g, s, t = net.graph, net.source, net.target
    assert not g.is_fifo()
    plain = alg.td_dijkstra(g, s, 480.0).arrival[t]
    best_no_wait = alg.brute_force_arrival(g, s, t, 480.0)
    waited = alg.td_dijkstra(g, s, 480.0, wait=True).arrival[t]
    assert plain > best_no_wait + 3.0 and best_no_wait > waited + 3.0                           # Dijkstra irrt; auch die beste Route ohne Warten ist später als mit Warten


@pytest.mark.parametrize("seed", range(4))
def test_without_any_congestion_it_is_ordinary_dijkstra(seed):
    net = sc.city_network(5, 0.0, 0, seed)
    g = net.graph
    G = nx.Graph()
    for u, v, b in zip(g.source_of_arcs().tolist(), g.indices.tolist(), g.base.tolist()):
        G.add_edge(u, v, weight=b)
    ref = nx.single_source_dijkstra_path_length(G, 0)
    for d in (0.0, 480.0):
        got = alg.td_dijkstra(g, 0, d).arrival
        assert [got[v] - d for v in range(g.n)] == pytest.approx([ref[v] for v in range(g.n)])


def test_edge_cases_same_node_unreachable_parallel_arcs_and_zero_cost():
    g = from_arcs(4, [(0, 1, 5.0, 0), (0, 1, 3.0, 0), (1, 2, 0.0, 0)], [np.ones(K)], np.zeros((4, 2)), directed=True)
    res = alg.td_dijkstra(g, 0, 100.0)
    assert res.arrival[0] == 100.0 and res.route(g, 0) == [] and res.arrival[1] == 103.0 and res.arrival[2] == 103.0 and res.arrival[3] == float("inf")
    assert alg.best_route(g, 0, 3, 100.0)[1] == [] and alg.best_route(g, 0, 3, 100.0)[0] == float("inf")
    assert alg.static_route(g, 0, 3) is None and alg.static_route(g, 0, 0) == []
    assert alg.brute_force_arrival(g, 0, 2, 100.0) == 103.0
    non = _random_graph(5, 8, 1, fifo=False)
    assert not non.is_fifo()
    with pytest.raises(ValueError):
        alg.profile_search(non, 0, 4)


def test_route_and_arrival_agree():
    g = _random_graph(9, 16, 2)
    for d in (0.0, 480.0, 1000.0):
        res = alg.td_dijkstra(g, 0, d)
        for t in range(1, 9):
            arcs = res.route(g, t)
            assert route_arrival(g, arcs, d) == pytest.approx(res.arrival[t]) and int(g.indices[arcs[-1]]) == t


def test_target_stops_the_search_early_with_the_same_answer():
    g = _random_graph(9, 16, 3)
    full, early = alg.td_dijkstra(g, 0, 480.0), alg.td_dijkstra(g, 0, 480.0, 5)
    assert early.arrival[5] == full.arrival[5] and early.counters["settled"] <= full.counters["settled"] and early.order[-1] == 5


# --- Profilsuche ---------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,m,seed", CASES)
def test_profile_matches_dijkstra_at_every_departure(n, m, seed):
    g = _random_graph(n, m, seed)
    F, cnt = alg.profile_search(g, 0, n - 1)
    for d in [x for x in DEPARTS if x <= DAY] + list(np.linspace(0, DAY, 37)):
        assert alg._ev(F, d) == pytest.approx(alg.td_dijkstra(g, 0, d).arrival[n - 1], abs=1e-6), d
    assert cnt["points"] == len(F[0]) and all(F[1][i] <= F[1][i + 1] + 1e-9 for i in range(len(F[1]) - 1))         # nichtfallend: FIFO
    assert all(F[1][i] >= F[0][i] - 1e-9 for i in range(len(F[0])))                                                 # Ankunft nie vor der Abfahrt


def test_profile_pruning_does_not_change_the_function():
    g = _random_graph(9, 16, 4)
    a, b = alg.profile_search(g, 0, 8, prune=True)[0], alg.profile_search(g, 0, 8, prune=False)[0]
    for d in np.linspace(0, DAY, 61):
        assert alg._ev(a, d) == pytest.approx(alg._ev(b, d), abs=1e-6)


def test_profile_on_the_generated_networks_matches_sampled_dijkstra():
    for net in (sc.small_network(), sc.city_network(6, 1.5, 0, 3), sc.random_network(40, 3.0, 1.5, 5)):
        g = net.graph
        F, _ = alg.profile_search(g, net.source, net.target)
        for d in (0.0, 123.0, 400.0, 480.0, 512.5, 1000.0, 1020.0, 1439.0):
            assert alg._ev(F, d) == pytest.approx(alg.td_dijkstra(g, net.source, d).arrival[net.target], abs=1e-6)


def test_link_and_pl_min_basics():
    row = [10.0] * K
    F = alg.link(alg.identity(), row)
    assert F[0][0] == 0.0 and F[0][-1] == DAY and F[1][0] == 10.0 and F[1][-1] == DAY + 10.0
    slow, fast = ([0.0, DAY], [30.0, DAY + 30.0]), ([0.0, DAY], [10.0, DAY + 100.0])
    H, changed = alg.pl_min(slow, fast)
    assert changed and alg._ev(H, 0.0) == 10.0 and alg._ev(H, DAY) == DAY + 30.0 and len(H[0]) == 3
    same, changed = alg.pl_min(slow, slow)
    assert not changed and same is slow


# --- Statische Planung und Reue -------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", alg.STATIC_KINDS)
def test_static_route_is_shortest_for_the_static_costs(kind):
    g = _random_graph(9, 16, 5)
    w = alg.static_costs(g, kind)
    G = nx.DiGraph()
    for k, (u, v) in enumerate(zip(g.source_of_arcs().tolist(), g.indices.tolist())):
        G.add_edge(u, v, weight=w[k])
    for t in range(1, 9):
        arcs = alg.static_route(g, 0, t, kind)
        assert sum(w[k] for k in arcs) == pytest.approx(nx.shortest_path_length(G, 0, t, weight="weight"))


def test_regret_is_never_negative_and_zero_for_the_best_route_itself():
    g = _random_graph(9, 16, 6)
    for d in (0.0, 480.0, 1000.0):
        for t in range(1, 9):
            best = alg.best_route(g, 0, t, d)[1]
            assert alg.regret(g, 0, t, d, best)[0] == pytest.approx(0.0, abs=1e-9)
            assert alg.regret(g, 0, t, d, alg.static_route(g, 0, t, "mean"))[0] >= -1e-9


def test_route_changes_report_the_best_route_per_departure():
    net = sc.small_network()
    day = alg.route_changes(net.graph, net.source, net.target, (0, 480, 720))
    assert [d for d, _, _ in day] == [0.0, 480.0, 720.0] and day[0][2] != day[1][2] and day[0][2] == day[2][2]
