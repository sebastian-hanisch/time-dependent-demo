"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-21, Toleranzen fangen Rundung ab). Fahrzeiten, Kantenprüfungen und Stützstellen sind plattformfest (reine Python-Rechnung mit festen Seeds);
Laufzeiten stehen in der App nur als Messwerte und werden hier nie geprüft."""

import pytest

import td_algorithm as alg
import td_constants as C
import td_evaluation as ev
import td_scenario as sc

PRESET = {"small": "🔀 Kleines Netz", "site": "🚧 Baustelle", "city": "🏙️ Stadtnetz", "random": "🕸️ Zufallsnetz"}


def _preset(key):
    p = dict(C.PRESETS[PRESET[key]])
    net, depart = p.pop("net"), p.pop("depart")
    a = ev.analyse(sc.make_network(net, **p))
    return a, ev.at(a, depart), depart


def _has(key, *needles):
    help_ = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in help_, (key, n)


# --- Preset-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_small_preset_numbers():
    a, m, depart = _preset("small")
    assert depart == 480 and len(a.segments) - 1 == 8 and m["routes_over_day"] == 3
    assert m["time"] == pytest.approx(18.32, abs=0.005) and m["static_time"] == pytest.approx(25.53, abs=0.005) and m["regret_rel"] == pytest.approx(0.393, abs=0.0005)
    assert ev.at(a, 180)["time"] == pytest.approx(10.0) and ev.at(a, 180)["static_time"] == pytest.approx(11.0)
    _has("small", "7 Orte", "achtmal", "drei verschiedenen", "18.3 min", "25.5 min", "39 %")


def test_site_preset_numbers():
    a, m, depart = _preset("site")
    assert depart == 480 and not a.fifo
    assert m["time"] == pytest.approx(58.89, abs=0.005) and m["wait_time"] == pytest.approx(50.0, abs=0.005) and m["wait_gain"] == pytest.approx(8.89, abs=0.005)
    _has("site", "6 Orte", "58.9 min", "50.0 min", "8.9 min", "8:30")


def test_city_preset_numbers():
    a, m, depart = _preset("city")
    assert depart == 480 and C.PRESETS[PRESET["city"]]["seed"] == 10 and a.net.graph.n == 64 and m["routes_over_day"] == 5
    assert m["time"] == pytest.approx(19.91, abs=0.005) and m["static_time"] == pytest.approx(21.67, abs=0.005) and m["regret_rel"] == pytest.approx(0.088, abs=0.0005)
    assert ev.at(a, 180)["time"] == pytest.approx(9.40, abs=0.005) and ev.at(a, 180)["static_time"] == pytest.approx(9.40, abs=0.005)
    _has("city", "8 × 8", "Seed 10", "fünf verschiedene", "19.9 min", "21.7 min", "+8.8 %", "9.4 min")


def test_random_preset_numbers():
    a, m, depart = _preset("random")
    assert depart == 480 and C.PRESETS[PRESET["random"]]["seed"] == 13 and a.net.graph.n == 60
    assert m["time"] == pytest.approx(27.08, abs=0.005) and m["static_time"] == pytest.approx(32.85, abs=0.006) and m["regret_rel"] == pytest.approx(0.213, abs=0.0005)
    _has("random", "60 Knoten", "Seed 13", "27.1 min", "32.8 min", "+21.3 %")


# --- Sidebar-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_strength_help_factors():
    p = sc.congestion_profile(1.5, 1.2)
    q = sc.congestion_profile(1.5, 0.3)
    assert p[16] == pytest.approx(2.8) and q[16] == pytest.approx(1.45) and p[34] == p[16]                          # 7:30-8:00 und 17:00-17:30 sind der Höchststau
    assert all(sc.rush_curve()[j] == 1.0 for j in (15, 16, 34, 35)) and sc.rush_curve()[17] < 1.0 and sc.rush_curve()[33] < 1.0             # 7:30, 8:00, 17:00, 17:30; 8:30 und 16:30 schon darunter


def test_sites_help_numbers():
    rows = {r["sites"]: r for r in ev.site_rows(sites=(10, 20, 40))}
    assert [rows[k]["share"] for k in (10, 20, 40)] == pytest.approx([0.010, 0.033, 0.117], abs=0.001)


# --- Experimente und die Tabelle "Wo die Annahmen enden" ----------------------------------------------------------------------------------------

def test_regret_experiment_numbers():
    rows = {r["strength"]: r for r in ev.regret_by_strength()}
    assert [rows[s]["changes"] for s in (0.5, 1.0, 1.5, 2.0, 3.0)] == pytest.approx([0.34, 0.66, 0.96, 1.0, 1.0], abs=0.005)
    assert [rows[s]["regret_480"] for s in (0.5, 1.0, 1.5, 2.0, 3.0)] == pytest.approx([0.0044, 0.0219, 0.0594, 0.1027, 0.1777], abs=0.0005)
    assert [rows[s]["regret_1020"] for s in (0.5, 1.0, 1.5, 2.0, 3.0)] == pytest.approx([0.0046, 0.0233, 0.0633, 0.1093, 0.1890], abs=0.0005)
    assert rows[1.5]["regret_180"] == pytest.approx(0.0049, abs=0.0005) and rows[1.5]["regret_720"] == pytest.approx(0.0049, abs=0.0005)                 # nachts und mittags: fast nichts
    assert rows[1.5]["regret_480"] > 10 * rows[1.5]["regret_180"]                                                                                      # die Mehrzeit entsteht in der Spitze
    rnd = ev.regret_by_strength("random", strengths=(1.5,), nodes=60)[0]
    assert rnd["changes"] == pytest.approx(0.38, abs=0.005) and rnd["regret_480"] == pytest.approx(0.0183, abs=0.0005)


def test_worst_case_regret_numbers():
    mean, free = ev.regret_worst(kind="mean"), ev.regret_worst(kind="freeflow")
    assert mean["mean"] == pytest.approx(0.0594, abs=0.0005) and mean["max"] == pytest.approx(0.276, abs=0.001) and mean["share_worse"] == pytest.approx(0.91, abs=0.005)
    assert free["mean"] == pytest.approx(0.0841, abs=0.0005) and free["max"] == pytest.approx(0.371, abs=0.001) and free["mean"] > mean["mean"]              # freie Fahrt planen ist schlechter als das Tagesmittel


def test_the_time_dependent_search_costs_no_extra_steps_but_the_profile_search_costs_a_lot():
    rows = {r["side"]: r for r in ev.cost_rows()}
    assert all(r["td_relaxed"] == r["static_relaxed"] and r["td_settled"] == r["static_settled"] for r in rows.values())
    assert [rows[s]["td_relaxed"] for s in (6, 8, 10, 12)] == [120, 224, 360, 528] and [rows[s]["n"] for s in (6, 8, 10, 12)] == [36, 64, 100, 144]
    assert [rows[s]["profile_points"] for s in (6, 8, 10, 12)] == pytest.approx([126.3, 199.7, 232.3, 289.7], abs=0.05)                              # Tabelle: 126 bis 290 Stützstellen
    assert [rows[s]["profile_work"] for s in (6, 12)] == pytest.approx([21409.3, 263789.7], abs=0.5)
    assert all(r["profile_work"] > 96 * r["td_relaxed"] for r in rows.values()) and rows[12]["profile_work"] > 5 * 96 * rows[12]["td_relaxed"]         # deutlich mehr als 96 einzelne Läufe


def test_dijkstra_errs_rarely_and_only_without_fifo():
    rows = {r["sites"]: r for r in ev.site_rows()}
    assert rows[0]["share"] == 0.0 and rows[5]["share"] == 0.0
    assert [rows[k]["share"] for k in (10, 20, 40)] == pytest.approx([0.0100, 0.0333, 0.1167], abs=0.0005)
    assert rows[40]["gain"] == pytest.approx(3.97, abs=0.005) and rows[40]["max_gain"] == pytest.approx(14.9, abs=0.05)                                     # Tabelle: 12 %, im Mittel 4.0 min
    assert rows[10]["gain"] < 0.2 and rows[20]["max_gain"] < 0.2                                                                                        # bei wenigen Baustellen ist der Fehler winzig


def test_the_preset_fifo_facts():
    assert sc.small_network().graph.is_fifo() and sc.city_network(8, 1.5, 0, 10).graph.is_fifo() and sc.random_network(60, 3.0, 1.5, 13).graph.is_fifo()
    assert not sc.site_network().graph.is_fifo()
    for key in ("small", "city", "random"):                                                               # die Profilsuche stimmt in jedem FIFO-Preset mit dem Dijkstra überein
        a, m, depart = _preset(key)
        F, _ = alg.profile_search(a.net.graph, a.net.source, a.net.target)
        assert alg._ev(F, depart) - depart == pytest.approx(m["time"], abs=1e-6), key
