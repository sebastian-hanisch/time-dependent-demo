"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Abfahrtszeiten, ausgeblendete Regler, Abspielen, Permalink, Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import td_constants as C
from td_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
VERDICT = {"🔀 Kleines Netz": "success", "🚧 Baustelle": "warning", "🏙️ Stadtnetz": "success", "🕸️ Zufallsnetz": "success"}


def _run(setup=None, timeout=600, net=None):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    if net:
        at.query_params["net"] = net
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input) + list(at.sidebar.select_slider)}


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()


def test_default_renders_without_exception_and_states_the_result():
    at = _run()
    assert any("Zeitabhängiges Routing in Aktion" in m.value for m in at.markdown)
    assert len(at.success) == 1 and "Die Uhrzeit ändert die Route" in at.success[0].value and not at.warning and not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_one_verdict(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    got = {"success": len(at.success), "warning": len(at.warning), "info": len(at.info)}
    assert got[VERDICT[name]] == 1 and sum(got.values()) == 1 and not at.error


@pytest.mark.parametrize("net", C.NETS)
@pytest.mark.parametrize("depart", (0, 480, 720, 1425))
def test_every_net_renders_at_night_in_the_rush_and_at_noon(net, depart):
    def setup(at):
        at.session_state["net_select"] = net
        at.session_state["depart_slider"] = depart
        at.session_state["side_slider"] = 5
    at = _run(setup)
    assert len(at.success) + len(at.warning) + len(at.info) == 1 and not at.error


def test_without_congestion_the_app_says_so():
    at = _run(lambda a: a.session_state.__setitem__("strength_slider", 0.0), net="city")
    assert len(at.info) == 1 and "Ohne Stau" in at.info[0].value


def test_the_same_route_at_night_is_explained_and_the_rush_hour_shows_the_difference():
    def night(a):
        _apply(a, C.PRESETS["🏙️ Stadtnetz"])
        a.session_state["depart_slider"] = 180
    at = _run(night)
    assert len(at.info) == 1 and "fahren die beste und die mit dem Tagesmittel geplante Route gleich" in at.info[0].value
    at.session_state["depart_slider"] = 480
    at.run()
    assert not at.exception and len(at.success) == 1


def test_the_baustelle_shows_the_gain_of_waiting_only_when_it_matters():
    at = _run(lambda a: _apply(a, C.PRESETS["🚧 Baustelle"]))
    assert "FIFO-Eigenschaft ist verletzt" in at.warning[0].value and "8.9 min weniger" in at.warning[0].value
    at.session_state["depart_slider"] = 900
    at.run()
    assert not at.exception and len(at.info) == 1 and "bringt Warten nichts" in at.info[0].value


def test_extreme_settings_render():
    def small(at):
        at.session_state["net_select"] = "city"
        at.session_state["side_slider"] = C.SIDE_MIN

    def big(at):
        at.session_state["net_select"] = "city"
        at.session_state["side_slider"] = C.SIDE_MAX
        at.session_state["sites_slider"] = C.SITES_MAX
        at.session_state["strength_slider"] = C.STRENGTH_MAX

    def flat(at):
        at.session_state["net_select"] = "random"
        at.session_state["nodes_slider"] = C.NODES_MIN
        at.session_state["degree_slider"] = C.DEGREE_MIN
    for setup in (small, big, flat):
        at = _run(setup)
        assert at.slider(key="td_step").value == at.slider(key="td_step").max


def test_hidden_controls_follow_the_net():
    common = {"Netz", "Stau-Stärke", "Abfahrtszeit"}
    small, site, city, rnd = (_labels(_run(net=n)) for n in ("small", "site", "city", "random"))
    assert small == site == common                                                                    # feste Aufgabe: kein Seed
    assert city == common | {"Kreuzungen je Seite", "Baustellen", "Zufalls-Seed"}
    assert rnd == common | {"Knoten", "Mittlerer Grad", "Zufalls-Seed"}


def test_hidden_slider_values_come_back_when_the_net_is_shown_again():
    # Die erste Sicht muss das Netz mit dem Regler sein: AppTest verliert den Wert, wenn der Regler zuerst ausgeblendet war (im echten Browser bleibt er erhalten).
    at = _run(net="city")
    at.session_state["sites_slider"] = 7
    at.run()
    at.session_state["net_select"] = "small"
    at.run()
    at.session_state["net_select"] = "city"
    at.run()
    assert not at.exception and at.slider(key="sites_slider").value == 7


def test_step_slider_returns_to_the_last_step_when_the_network_or_the_time_changes():
    at = _run(net="city")
    at.slider(key="td_step").set_value(5)
    at.run()
    assert at.slider(key="td_step").value == 5
    at.session_state["depart_slider"] = 600
    at.run()
    assert not at.exception and at.slider(key="td_step").value == at.slider(key="td_step").max


def test_play_renders_several_frames_without_duplicate_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    for setup in (lambda a: None, lambda a: a.session_state.__setitem__("net_select", "city"), lambda a: a.session_state.__setitem__("net_select", "random"), lambda a: a.session_state.__setitem__("net_select", "site")):
        at = _run(setup)
        _play(at)
        assert not at.exception, [e.value for e in at.exception]


def test_permalink_parameters_select_the_net_and_are_clamped_and_snapped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "city"
    at.query_params["sites"] = "999"
    at.query_params["depart"] = "487"
    at.query_params["strength"] = "9"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == "city"
    assert at.slider(key="sites_slider").value == C.SITES_MAX and at.select_slider(key="depart_slider").value == 480 and at.slider(key="strength_slider").value == C.STRENGTH_MAX


def test_unknown_net_in_the_permalink_falls_back_to_the_default():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET


def test_experiments_run_on_demand():
    at = _run()
    assert not any("Mittel über 5 Netze" in c.value for c in at.caption)
    for key in ("regret_start", "cost_start", "sites_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, (key, [e.value for e in at.exception])
    text = " ".join(c.value for c in at.caption)
    for needle in ("die Mehrzeit entsteht in der Spitze", "prüft **genau dieselben Kanten** wie das statische", "Dijkstra irrt also **selten**"):
        assert needle in text, needle


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    # Karte und Kurve stehen in der Play-Schleife: ihre Schlüssel tragen den Schritt
    assert sorted(keys) == sorted(["net_chart", "curve_chart", "regret_chart", "cost_chart", "sites_chart"]), keys
    assert sum('key=f"' in c for c in calls) == 2 and all('_{current}"' in c for c in calls if 'key=f"' in c)
    viz = (ROOT / "td_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("_base(fig") >= 5


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))
