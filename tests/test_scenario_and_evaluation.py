"""Netze (kleines Netz, Baustelle, Stadtnetz mit Hauptachsen, Zufallsnetz), Kennzahlen, Tageslauf, Bildfolge und Abbildungen, Messreihen."""

import networkx as nx
import numpy as np
import pytest

import td_algorithm as alg
import td_constants as C
import td_evaluation as ev
import td_scenario as sc
import td_visualization as viz
from td_graph import K, route_arcs_nodes


def _connected(g):
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from((int(u), int(v)) for u, v in zip(g.source_of_arcs(), g.indices))
    return nx.is_connected(G)


# --- Netze ---------------------------------------------------------------------------------------------------------------------------------------

def test_the_rush_curve_has_two_peaks_and_the_profiles_start_at_one():
    r = sc.rush_curve()
    assert len(r) == K and r[0] == 0.0 and r[-1] == 0.0 and r[16] == 1.0 and r[34] == 1.0 and r.max() == 1.0 and r[24] < 0.2         # 8:00, 17:00, 12:00
    p = sc.congestion_profile(1.5, 1.2)
    assert p[0] == 1.0 and p[16] == pytest.approx(2.8) and p[24] == pytest.approx(1.0 + 1.5 * 1.2 * r[24])
    assert (sc.congestion_profile(0.0, 1.2) == 1.0).all()


def test_the_site_profile_is_steeper_than_the_clock_and_ends_at_half_past_eight():
    p = sc.site_profile(20.0)
    assert p[16] == pytest.approx(3.25) and p[17] == pytest.approx(1.0) and p[0] == pytest.approx(3.25) and p[-1] == pytest.approx(1.0)              # 45 Minuten Zuschlag auf 20 Minuten
    row = 20.0 * p
    assert (row[17] - row[16]) / 30.0 == pytest.approx(-1.5)                                          # Fahrzeit sinkt um 1.5 Minuten je Minute


def test_small_network_changes_its_best_route_three_times_over_the_day():
    net = sc.small_network()
    g = net.graph
    assert g.n == 7 and g.is_fifo() and (g.names[net.source], g.names[net.target]) == ("Start", "Ziel")
    a = ev.analyse(net)
    routes = {tuple(g.names[v] for v in route_arcs_nodes(g, net.source, arcs)) for _, _, arcs in a.day}
    assert routes == {("Start", "Ring", "Ziel"), ("Start", "Markt", "Ziel"), ("Start", "Dorf", "Ziel")}
    assert [g.names[v] for v in route_arcs_nodes(g, net.source, a.static_arcs["mean"])] == ["Start", "Markt", "Ziel"]      # die Durchschnittskarte wählt die Route, die nur zwischendurch die beste ist


def test_site_network_is_not_fifo_and_only_the_site_arc_breaks_it():
    net = sc.site_network()
    g = net.graph
    assert not g.is_fifo() and set(np.flatnonzero(~g.fifo_arcs())) == set(net.site_arcs) and len(net.site_arcs) == 2


@pytest.mark.parametrize("seed", range(3))
def test_city_network_is_a_connected_grid_with_faster_arterials(seed):
    net = sc.city_network(8, 1.5, 0, seed)
    g = net.graph
    assert g.n == 64 and g.m == 2 * 2 * 8 * 7 and _connected(g) and g.is_fifo() and len(net.arterial) == g.m and (net.source, net.target) == (0, 63)
    art = np.array(net.arterial)
    assert g.base[art].mean() < g.base[~art].mean()
    # im Höchststau ist die Hauptachse je Meter langsamer als bei freier Fahrt und stärker betroffen als die Nebenstraße
    assert g.tab[art][:, 16].mean() / g.base[art].mean() > g.tab[~art][:, 16].mean() / g.base[~art].mean() > 1.0


def test_city_sites_break_fifo_only_on_local_streets():
    assert sc.city_network(8, 1.5, 0, 3).graph.is_fifo() and not sc.city_network(8, 1.5, 5, 3).graph.is_fifo()
    net = sc.city_network(8, 1.5, 5, 3)
    assert len(net.site_arcs) == 10 and not any(net.arterial[k] for k in net.site_arcs)                    # 5 Baustellen, jede in beiden Richtungen
    assert len(sc.city_network(8, 1.5, 999, 3).site_arcs) == 2 * int(sum(1 for a in sc.city_network(8, 1.5, 0, 3).arterial if not a) / 2)   # nie mehr Baustellen als Nebenstraßen


def test_random_network_is_connected_fifo_and_has_fast_roads():
    net = sc.random_network(80, 3.0, 1.5, 4)
    g = net.graph
    assert _connected(g) and g.is_fifo() and abs(g.m / g.n - 3.0) < 0.1 and not net.geometric and net.target != net.source
    assert set(g.prof.tolist()) == {0, 1}


def test_make_network_rejects_unknown_nets():
    with pytest.raises(ValueError):
        sc.make_network("ring")


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", C.NETS)
def test_analysis_invariants_for_every_net(key):
    net = sc.make_network(key, side=6, nodes=40)
    a = ev.analyse(net)
    assert a.fifo == net.graph.is_fifo() and a.segments[0][0] == 0 and a.segments[-1][1] == 1440
    assert all(a.segments[i][1] == a.segments[i + 1][0] for i in range(len(a.segments) - 1))               # lückenlos
    assert len(a.curve_x) == len(a.curve["best"]) == len(a.curve["mean"]) == len(a.curve["freeflow"]) == 145
    best = a.curve["best"]
    assert all(m >= b - 1e-6 for m, b in zip(a.curve["mean"], best)) and all(f >= b - 1e-6 for f, b in zip(a.curve["freeflow"], best))          # die beste Route ist nie schlechter als eine feste
    if a.fifo:
        assert a.profile_points > 0 and "wait" not in a.curve
    else:
        assert all(w <= b + 1e-6 for w, b in zip(a.curve["wait"], best))
    for depart in (0, 480, 1020):
        m = ev.at(a, depart)
        assert m["reachable"] and m["time"] == pytest.approx(a.curve["best"][depart // 10], abs=1e-6)              # die Kurve ist an dieser Stelle die Dijkstra-Antwort
        assert m["regret"] >= -1e-9 and m["static_time"] >= m["time"] - 1e-9 and (not m["same_route"] or m["regret"] == pytest.approx(0.0, abs=1e-9))
        assert m["settled"] <= net.graph.n and m["relaxed"] >= m["settled"] - 1 and m["routes_over_day"] == len({arcs for _, _, arcs in a.day})


def test_profile_curve_equals_sampled_dijkstra_on_the_curve_grid():
    net = sc.city_network(6, 1.5, 0, 3)
    a = ev.analyse(net)
    for i in (0, 10, 48, 50, 102, 144):
        x = a.curve_x[i]
        assert a.curve["best"][i] == pytest.approx(alg.best_route(net.graph, net.source, net.target, x)[0] - x, abs=1e-6)


def test_at_reports_waiting_only_without_fifo():
    assert "wait_gain" not in ev.at(ev.analyse(sc.small_network()), 480)
    m = ev.at(ev.analyse(sc.site_network()), 480)
    assert m["wait_gain"] > 5.0 and m["wait_time"] < m["time"]


def test_without_congestion_the_static_plan_is_never_wrong():
    net = sc.city_network(6, 0.0, 0, 2)
    a = ev.analyse(net)
    for depart in (0, 480, 1020):
        m = ev.at(a, depart)
        assert m["regret"] == pytest.approx(0.0, abs=1e-9)
    assert len(a.segments) == 1


def test_frames_cover_the_settled_nodes():
    order = list(range(200))
    f = ev.frames(None, order)
    assert f[0] == 0 and f[-1] == 200 and len(f) <= 61 and f == sorted(set(f))
    assert ev.frames(None, list(range(7))) == list(range(8))


# --- Abbildungen -------------------------------------------------------------------------------------------------------------------------------------

def test_charts_render_for_every_net_and_every_step():
    for key in C.NETS:
        net = sc.make_network(key, side=6, nodes=40, sites=3)
        a = ev.analyse(net)
        m = ev.at(a, 480)
        res = alg.td_dijkstra(net.graph, net.source, 480)
        frames = ev.frames(net.graph, res.order)
        for k in (frames[0], frames[len(frames) // 2], frames[-1]):
            routes = [(m["arcs"], "beste", "#000", "solid"), (m["static_arcs"], "statisch", "#f00", "dash")] if k == frames[-1] else []
            viz.build_network(net, res, k, 480, routes)
        viz.build_curve(a, 480)
        if net.graph.names:
            assert viz.route_table(a)
    viz.build_regret([{"strength": 0.5, "regret_480": 0.01, "regret_1020": 0.01, "regret_180": 0.0, "regret_720": 0.0}])
    viz.build_cost([{"n": 36, "td_relaxed": 120, "profile_work": 5000}])
    viz.build_sites([{"sites": 0, "share": 0.0}])


def test_route_table_lists_the_day_of_the_small_network():
    rows = viz.route_table(ev.analyse(sc.small_network()))
    assert rows[0]["Abfahrt"].startswith("0:00–6:15") and rows[-1]["Abfahrt"].endswith("24:00") and rows[0]["Beste Route"] == "Start → Ring → Ziel" and len(rows) == 9


def test_hhmm_formats_times_of_day():
    assert C.hhmm(0) == "0:00" and C.hhmm(485) == "8:05" and C.hhmm(1440) == "0:00" and C.hhmm(1015) == "16:55"


# --- Messreihen ---------------------------------------------------------------------------------------------------------------------------------------

def test_pick_pairs_are_far_apart_and_reproducible():
    net = sc.city_network(8, 1.5, 0, 3)
    pairs = ev.pick_pairs(net, 10, 3)
    assert pairs == ev.pick_pairs(net, 10, 3) and len(pairs) == 10 and all(s != t for s, t in pairs)


def test_regret_rows_grow_with_the_strength_and_vanish_without_it():
    rows = ev.regret_by_strength(strengths=(0.0, 1.0, 2.0), seeds=C.SWEEP_SEEDS[:2], pairs=8)
    assert rows[0]["regret_480"] == pytest.approx(0.0, abs=1e-9) and rows[0]["changes"] == 0.0
    assert rows[0]["regret_480"] < rows[1]["regret_480"] < rows[2]["regret_480"] and rows[2]["regret_480"] > rows[2]["regret_180"]


def test_cost_rows_show_equal_effort_for_static_and_time_dependent_dijkstra():
    rows = ev.cost_rows(sides=(5, 6), seeds=C.SWEEP_SEEDS[:2])
    assert all(r["td_relaxed"] == r["static_relaxed"] and r["td_settled"] == r["static_settled"] for r in rows) and rows[0]["profile_work"] < rows[1]["profile_work"]


def test_site_rows_have_no_case_without_sites_and_grow_with_them():
    rows = ev.site_rows(sites=(0, 30), seeds=C.SWEEP_SEEDS[:2], pairs=8)
    assert rows[0]["share"] == 0.0 and rows[1]["share"] > 0.0 and rows[1]["gain"] > 0.0
