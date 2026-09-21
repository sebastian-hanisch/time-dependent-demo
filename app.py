"""Zeitabhängiges Routing - die Uhrzeit ändert die Route - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - das zeitabhängige Dijkstra samt Profilsuche - und lässt stattdessen das Beispiel wachsen.
Neuntes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe, Ast von Dijkstra: die Kosten einer Kante hängen von der Abfahrtszeit ab (Stau), und Dijkstra bleibt genau dann korrekt, wenn niemand durch späteres Losfahren früher ankommt (FIFO).
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import td_algorithm as alg
import td_constants as C
import td_evaluation as ev
from td_graph import route_arcs_nodes
from td_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from td_scenario import make_network
from td_visualization import build_cost, build_curve, build_network, build_regret, build_sites, route_table

st.set_page_config(page_title="Zeitabhängiges Routing – Sebastian Hanisch", layout="wide")


def _num(x):
    return f"{x:,.0f}".replace(",", ".")


def _pct(x, digits=0):
    return "–" if x is None or np.isnan(x) else f"{x:.{digits}%}"


@st.cache_resource(show_spinner=False, max_entries=8)
def _network(params):
    return make_network(*params)


@st.cache_resource(show_spinner=False, max_entries=8)
def _analysis(params):
    return ev.analyse(_network(params))


@st.cache_data(show_spinner=False)
def _regret():
    return ev.regret_by_strength(), ev.regret_by_strength("random", strengths=(1.5,), nodes=60)[0], ev.regret_worst(kind="mean"), ev.regret_worst(kind="freeflow")


@st.cache_data(show_spinner=False)
def _cost():
    return ev.cost_rows()


@st.cache_data(show_spinner=False)
def _sites():
    return ev.site_rows()


st.title("🕗 Zeitabhängiges Routing – die Uhrzeit ändert die Route")
st.markdown(
    """
Ein Navigationsgerät rechnet mit einer Karte, auf der jede Straße eine feste Fahrzeit hat. Im echten Verkehr hängt die Fahrzeit einer Straße aber **von der Uhrzeit ab**: die schnelle Hauptachse ist morgens um 8 Uhr verstopft, die Landstraße nicht.
Beim **zeitabhängigen Dijkstra** ist das Label eines Knotens die **Ankunftszeit**, und jede Kante wird zu dem Zeitpunkt ausgewertet, zu dem man an ihrem Anfang ankommt. Der Rechenaufwand bleibt derselbe wie beim gewöhnlichen Dijkstra - die Frage ist, **wann das noch stimmt**:
genau dann, wenn niemand durch **späteres Losfahren früher ankommt** (FIFO-Eigenschaft). Eine **Profilsuche** liefert außerdem die Ankunftszeit als Funktion der Abfahrtszeit für den ganzen Tag.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - neuntes Stück der Kürzeste-Wege-Linie der \"Konzepte\"-Reihe, Ast von Dijkstra - **ein** Verfahren an einem wachsenden Beispiel. "
    "Die Schwäche des Verfahrens: es kennt den Stau im Voraus (eine Prognose, keine Live-Daten), und ohne FIFO-Eigenschaft ist es nicht mehr korrekt. Das Verfahren geht auf Cooke und Halsey (1966) und Dreyfus (1969) zurück, die FIFO-Bedingung auf Orda und Rom (1990); "
    "alle Netze, Stauverläufe und Zahlen dieser Demo sind eigene Konstruktionen und Messungen."
)

with st.expander("So funktioniert zeitabhängiges Routing", expanded=True):
    st.markdown(
        """
1. **Fahrzeit als Funktion:** jede Kante hat eine Grundfahrzeit (freie Fahrt) und einen Stauverlauf; ihre Fahrzeit $\\tau_e(t)$ bei Abfahrt zur Zeit $t$ ist stückweise linear zwischen Stützstellen alle 30 Minuten. Die Ankunft ist $t + \\tau_e(t)$.
2. **Zeitabhängiges Dijkstra:** das Label eines Knotens ist seine früheste Ankunftszeit. Wird ein Knoten $u$ mit der Ankunft $a_u$ festgelegt, ist die Ankunft an seinem Nachbarn $v$ höchstens $a_u + \\tau_{uv}(a_u)$.
3. **FIFO-Eigenschaft:** die Ankunft $t + \\tau_e(t)$ fällt nirgends, die Fahrzeit sinkt also nie schneller als eine Minute je Minute. Dann ist früher ankommen an einem Knoten nie schlechter, und Dijkstra bleibt korrekt.
4. **Profilsuche:** statt einer Zahl trägt jeder Knoten eine **Funktion** (Ankunft in Abhängigkeit von der Abfahrt); Kanten verknüpfen sie, das Minimum zweier Funktionen wählt die bessere Route. Ergebnis: Fahrzeit gegen Abfahrtszeit für den ganzen Tag.
5. **Statische Planung:** wer mit festen Kosten plant (freie Fahrt oder Tagesmittel), fährt die Route, die zu keiner Uhrzeit die beste sein muss - die **Mehrzeit** gegen die beste Route heißt hier Reue.
6. **Ohne FIFO:** endet eine Baustelle, kann sich Warten lohnen; Dijkstra kann dann eine zu späte Ankunft melden. Darf man warten, gilt FIFO für die "Kante mit Warten" wieder.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", C.NETS, key="net_select", format_func=lambda k: C.NET_LABELS[k],
        help="Klein und fest (die beste Route wechselt mit der Uhrzeit; Baustelle mit Überholen) oder erzeugt (Stadtnetz mit Hauptachsen, Zufallsnetz). Fahrzeiten in Minuten, Uhrzeiten in Minuten seit Mitternacht.",
    )
    strength = st.slider("Stau-Stärke", *bounds("strength_slider"), key="strength_slider", step=0.25,
                         help="0 = kein Stau (feste Fahrzeiten). Im Höchststau (morgens 7:30–8:00, abends 17:00–17:30; davor und danach klingt er ab) wird die Fahrzeit einer Hauptachse mit Stärke 1.5 auf das 2.8-Fache, die einer Nebenstraße auf das 1.45-Fache; die Stärke skaliert beides. Mit 0 ist die zeitabhängige Suche überflüssig.")
    depart = st.select_slider("Abfahrtszeit", C.DEPART_OPTIONS, key="depart_slider", format_func=C.hhmm,
                              help="Wann man am Start losfährt (Schritte von 15 Minuten). Die Karte, die Routen und alle Kennzahlen gelten für diese Uhrzeit; das Diagramm zeigt den ganzen Tag.")
    if net_key == "city":
        side = st.slider("Kreuzungen je Seite", *bounds("side_slider"), key="side_slider", help="Größe des Rasters: n = Seite² Knoten. Die Routen wechseln im Lauf des Tages, wenn Start und Ziel weit auseinander liegen.")
        st.session_state[KEPT["side_slider"]] = side
        sites = st.slider("Baustellen", *bounds("sites_slider"), key="sites_slider",
                          help="So viele Nebenstraßen bekommen eine Baustelle, die um 8:30 endet (45 Minuten Zuschlag davor): dort wird die FIFO-Eigenschaft verletzt. Bei 10 / 20 / 40 Baustellen im 8 × 8-Netz lohnt sich Warten in 1 % / 3 % / 12 % der Abfragen.")
        st.session_state[KEPT["sites_slider"]] = sites
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
        sites = int(st.session_state.get(KEPT["sites_slider"], C.DEFAULT_SITES))
    if net_key == "random":
        nodes = st.slider("Knoten", *bounds("nodes_slider"), key="nodes_slider", step=10, help="Anzahl der Knoten n.")
        st.session_state[KEPT["nodes_slider"]] = nodes
        degree = st.slider("Mittlerer Grad", *bounds("degree_slider"), key="degree_slider", step=0.5, help="Kanten je Knoten.")
        st.session_state[KEPT["degree_slider"]] = degree
    else:
        nodes = int(st.session_state.get(KEPT["nodes_slider"], C.DEFAULT_NODES))
        degree = float(st.session_state.get(KEPT["degree_slider"], C.DEFAULT_DEGREE))
    if net_key in ("city", "random"):
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für das Netz.")
    else:
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen.")

# nicht zum Netz gehörende Regler ändern das Netz nicht: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
d = dict(side=C.DEFAULT_SIDE, sites=C.DEFAULT_SITES, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, seed=C.DEFAULT_SEED)
if net_key == "city":
    d.update(side=int(side), sites=int(sites), seed=int(seed))
elif net_key == "random":
    d.update(nodes=int(nodes), degree=round(float(degree), 1), seed=int(seed))
strength = round(float(strength), 2)
params = (net_key, strength, d["side"], d["sites"], d["nodes"], d["degree"], d["seed"])
sync_query_params({"net_select": net_key, "strength_slider": strength, "depart_slider": int(depart), "side_slider": int(side), "sites_slider": int(sites), "nodes_slider": int(nodes),
                   "degree_slider": round(float(degree), 1), "seed_input": int(seed)})
with st.spinner("Rechne ..."):
    a = _analysis(params)
net, g = a.net, a.net.graph
small = bool(g.names)
m = ev.at(a, depart)
res = alg.td_dijkstra(g, net.source, depart)
frame_list = ev.frames(g, res.order)
last_step = len(frame_list) - 1
view_key = (params, int(depart))
if st.session_state.get("td_step_owner") != view_key:
    st.session_state["td_step_owner"] = view_key
    st.session_state["td_step"] = last_step


def _names(arcs):
    return " → ".join(g.names[v] for v in route_arcs_nodes(g, net.source, arcs))


def _route_text(arcs):
    return _names(arcs) if small else f"{len(arcs)} Kanten"


# --- Zeitabhängiges Routing in Aktion ------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Zeitabhängiges Routing in Aktion")
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.slider("Festgelegte Knoten", 0, last_step, key="td_step",
                     help="Wie viele Knoten das zeitabhängige Dijkstra schon festgelegt hat (in der Reihenfolge der Ankunftszeit): 0 = noch keiner, ganz rechts = alle. Die Farbe zeigt die Minuten seit der Abfahrt. Bei vielen Knoten zeigt die Ansicht etwa 60 Zwischenstände.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _render(current):
    k = frame_list[current]
    final = k >= len(res.order)
    with view_slot.container():
        routes = []
        if final and m["reachable"]:
            routes = [(m["arcs"], "beste Route", C.COLORS["td"], "solid")]
            if not m["same_route"]:
                routes.append((m["static_arcs"], "mit Tagesmittel geplant", C.COLORS["static"], "dash"))
            if not a.fifo and m["wait_gain"] > 1e-6:
                routes.append((m["wait_arcs"], "beste Route mit Warten", C.COLORS["wait"], "dot"))
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_network(net, res, k, depart, routes, height=400 if small else 460), width="stretch", key=f"net_chart_{current}")
        c2.plotly_chart(build_curve(a, depart), width="stretch", key=f"curve_chart_{current}")
        if small and final:
            st.markdown(f"**Tageslauf der besten Route** (Fahrzeiten an den Kanten: Abfahrt {C.hhmm(depart)})")
            st.dataframe(pd.DataFrame(route_table(a)), hide_index=True, width="stretch")
        if final and m["reachable"]:
            st.markdown(f"**Beste Route bei Abfahrt {C.hhmm(depart)}:** {m['time']:.1f} min, Ankunft {C.hhmm(m['arrival'])} - {_route_text(m['arcs'])}.")


if auto_play:
    n_frames = min(last_step + 1, 40)
    for kk in sorted({int(round(x)) for x in np.linspace(0, last_step, n_frames)}):
        _render(kk)
        time.sleep(min(0.7, 6.0 / n_frames))
    step = last_step
else:
    _render(step)
st.caption(net.note)

st.markdown("---")

# --- Die Uhrzeit ändert die Route ------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Uhrzeit ändert die Route")
st.caption("**Zähler** = Schritte, die ein Verfahren ausführt: festgelegte Knoten, Kantenprüfungen, verarbeitete Stützstellen. Sie sind plattformfest. Laufzeiten stehen nur als Messwerte im Vergleich unten.")
if not m["reachable"]:
    st.warning("⚠️ Das Ziel ist vom Start aus nicht erreichbar.")
else:
    changes = len(a.segments) - 1
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Beste Route", f"{m['time']:.1f} min", delta=f"Ankunft {C.hhmm(m['arrival'])}", delta_color="off", help="Fahrzeit der zeitabhängig besten Route bei dieser Abfahrtszeit.")
    if m["same_route"]:
        p2.metric("Mit Tagesmittel geplant", f"{m['static_time']:.1f} min", delta="gleiche Route", delta_color="off", help="Route, die mit den Tagesmittel-Kosten am kürzesten ist, zu dieser Uhrzeit gefahren.")
    else:
        p2.metric("Mit Tagesmittel geplant", f"{m['static_time']:.1f} min", delta=f"+{m['regret_rel']:.0%} gegen die beste", delta_color="off", help="Route, die mit den Tagesmittel-Kosten am kürzesten ist, zu dieser Uhrzeit gefahren; die Mehrzeit gegen die beste Route ist die Reue.")
    p3.metric("Beste Routen im Tagesverlauf", f"{m['routes_over_day']}", delta=f"{changes} Wechsel", delta_color="off", help="Wie viele verschiedene Routen zu irgendeiner Abfahrtszeit (alle 15 Minuten geprüft) die beste sind, und wie oft sie im Lauf des Tages wechselt.")
    p4.metric("Kantenprüfungen", _num(m["relaxed"]), delta=f"statisch: {_num(m['static_relaxed'])}", delta_color="off", help="Kanten, die das zeitabhängige Dijkstra bis zum Ziel geprüft hat, gegen das statische Dijkstra: dieselbe Zahl, nur die Auswertung der Kante ist aufwendiger.")
    if not a.fifo:
        if m["wait_gain"] > 1e-6:
            st.warning(f"⚠️ **Die FIFO-Eigenschaft ist verletzt:** Dijkstra ohne Warten meldet {m['time']:.1f} min (Ankunft {C.hhmm(m['arrival'])}), mit erlaubtem Warten sind es {m['wait_time']:.1f} min (Ankunft {C.hhmm(m['arrival'] - m['wait_gain'])}) - "
                       f"{m['wait_gain']:.1f} min weniger. Die Baustelle endet, und wer später ankommt, kommt schneller durch.")
        else:
            st.info("ℹ️ In diesem Netz ist die FIFO-Eigenschaft verletzt (Baustelle), aber bei dieser Abfahrtszeit bringt Warten nichts. Probieren Sie eine Abfahrtszeit um 8:00 (Netz \"Baustelle\").")
    elif strength == 0:
        st.info("ℹ️ Ohne Stau sind alle Fahrzeiten konstant: die zeitabhängige Suche liefert dieselbe Route wie das statische Dijkstra. Erhöhen Sie die Stau-Stärke.")
    elif m["same_route"]:
        st.info(f"ℹ️ Zu dieser Uhrzeit fahren die beste und die mit dem Tagesmittel geplante Route gleich. Im Lauf des Tages ist aber jede von {m['routes_over_day']} Routen einmal die beste - wählen Sie eine andere Abfahrtszeit (z. B. 8:00).")
    else:
        st.success(f"✅ **Die Uhrzeit ändert die Route:** um {C.hhmm(depart)} braucht die beste Route {m['time']:.1f} min, die mit den Tagesmittel-Kosten geplante {m['static_time']:.1f} min ({m['regret_rel']:+.0%}); mit freier Fahrt geplant sind es {m['freeflow_time']:.1f} min. "
                   f"Im Lauf des Tages sind {m['routes_over_day']} verschiedene Routen die beste.")

st.markdown("---")

# --- Vergleich -------------------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – statisch, zeitabhängig und Profilsuche im Vergleich"):
    t0 = time.perf_counter()
    alg.static_dijkstra(g, net.source, alg.static_costs(g, "mean"), net.target)
    t_static = time.perf_counter() - t0
    t0 = time.perf_counter()
    alg.td_dijkstra(g, net.source, depart, net.target)
    t_td = time.perf_counter() - t0
    if m["reachable"]:
        rel = m["relaxed"]
        work = a.profile_counters.get("points_processed") if a.fifo else None
        st.table({"Verfahren": ["Statisches Dijkstra (Tagesmittel)", "Zeitabhängiges Dijkstra (eine Abfahrtszeit)", "96 zeitabhängige Läufe (alle 15 Minuten)", "Profilsuche (ganzer Tag, exakt)"],
                  "Zähler": [f"{_num(m['static_relaxed'])} Kantenprüfungen", f"{_num(rel)} Kantenprüfungen", f"{_num(96 * rel)} Kantenprüfungen", f"{_num(work)} Stützstellen" if work else "nur mit FIFO"],
                  "Laufzeit [ms]": [f"{t_static * 1000:.2f}", f"{t_td * 1000:.2f}", "–", f"{a.seconds['profile'] * 1000:.1f}" if a.fifo else "–"]})
    st.caption("Die Laufzeiten sind Messwerte dieses Laufs (reines Python, ein Lauf, schwankend). Das zeitabhängige Dijkstra prüft dieselben Kanten wie das statische - nur die Auswertung einer Kante ist eine Interpolation statt einer Zahl. Die Profilsuche liefert die exakte Ankunftsfunktion "
               "(eine Funktion je Knoten) und ist nur bei FIFO möglich; sie verarbeitet deutlich mehr Stützstellen, als 96 einzelne Abfragen Kanten prüfen (siehe Experiment zum Aufwand).")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie viel kostet die statische Planung?")
if st.button("Mehrzeit gegen Stau-Stärke und Uhrzeit messen (dauert einen Moment)", key="regret_start"):
    st.session_state["regret_on"] = True
if st.session_state.get("regret_on"):
    with st.spinner("Rechne 5 Stärken × 5 Netze × 20 Paare ..."):
        rrows, rrand, worst_mean, worst_free = _regret()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_regret(rrows), width="stretch", key="regret_chart")
    c2.table({"Stärke": [f"{r['strength']:g}" for r in rrows], "8:00": [f"+{r['regret_480']:.1%}" for r in rrows], "Route wechselt": [f"{r['changes']:.0%}" for r in rrows]})
    r15 = rrows[2]
    st.caption(f"Stadtnetz 8 × 8, Mittel über 5 Netze × 20 zufällige Paare (mindestens 8 Kanten Abstand); geplant mit den Tagesmittel-Kosten, gefahren zur Uhrzeit. Bei Stärke 1.5 kostet das um 8:00 im Mittel {r15['regret_480']:.1%} mehr Fahrzeit (Höchstwert {worst_mean['max']:.0%}), "
               f"nachts (3:00) und mittags (12:00) nur {r15['regret_180']:.1%}: die Mehrzeit entsteht in der Spitze. Mit den Kosten der freien Fahrt geplant sind es um 8:00 im Mittel {worst_free['mean']:.1%} (Höchstwert {worst_free['max']:.0%}). "
               f"Bei Stärke 1.5 ist die beste Route um 8:00 in {r15['changes']:.0%} der Paare eine andere als um 3:00; im Zufallsnetz (60 Knoten) sind es {rrand['changes']:.0%} und die Mehrzeit {rrand['regret_480']:.1%}.")

st.markdown("---")

st.subheader("🔬 Was kostet die Zeitabhängigkeit?")
if st.button("Aufwand gegen Netzgröße messen (dauert einen Moment)", key="cost_start"):
    st.session_state["cost_on"] = True
if st.session_state.get("cost_on"):
    with st.spinner("Rechne 4 Größen × 3 Netze ..."):
        crows = _cost()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_cost(crows), width="stretch", key="cost_chart")
    c2.table({"Knoten": [_num(r["n"]) for r in crows], "Dijkstra": [_num(r["td_relaxed"]) for r in crows], "Profil": [_num(r["profile_work"]) for r in crows]})
    lo, hi = crows[0], crows[-1]
    st.caption(f"Stadtnetz mit {_num(lo['n'])} bis {_num(hi['n'])} Knoten, Mittel über 3 Netze. Das zeitabhängige Dijkstra prüft **genau dieselben Kanten** wie das statische ({_num(hi['td_relaxed'])} bei {_num(hi['n'])} Knoten) - die Zeitabhängigkeit kostet keine zusätzlichen Schritte. "
               f"Die Profilsuche liefert dafür den ganzen Tag: sie verarbeitet {_num(lo['profile_work'])} bis {_num(hi['profile_work'])} Stützstellen, das ist ein Mehrfaches von 96 einzelnen Läufen ({_num(96 * lo['td_relaxed'])} bis {_num(96 * hi['td_relaxed'])} Kantenprüfungen); "
               f"die Ankunftsfunktion am Ziel wächst von {lo['profile_points']:.0f} auf {hi['profile_points']:.0f} Stützstellen. Ihr Vorteil ist nicht der Aufwand, sondern dass sie **exakt** ist: keine Abfahrtszeit fällt durch das Raster.")

st.markdown("---")

st.subheader("🔬 Wann irrt Dijkstra?")
if st.button("Baustellen gegen Fehlerquote messen (dauert einen Moment)", key="sites_start"):
    st.session_state["sites_on"] = True
if st.session_state.get("sites_on"):
    with st.spinner("Rechne 5 Baustellenzahlen × 5 Netze × 15 Paare × 4 Abfahrtszeiten ..."):
        srows = _sites()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_sites(srows), width="stretch", key="sites_chart")
    c2.table({"Baustellen": [str(r["sites"]) for r in srows], "Warten hilft": [f"{r['share']:.1%}" for r in srows], "Mehrzeit [min]": [f"{r['gain']:.1f}" for r in srows]})
    st.caption(f"Stadtnetz 8 × 8, Abfahrt um 7:00, 7:30, 8:00 oder 8:20 (Baustellen enden um 8:30, 45 Minuten Zuschlag), Mittel über 5 Netze × 15 Paare. Ohne Baustellen (FIFO) gibt es keinen einzigen Fall. Mit 10 Baustellen lohnt sich Warten in {srows[2]['share']:.1%} der Abfragen, mit 20 in {srows[3]['share']:.1%}, "
               f"mit 40 in {srows[4]['share']:.1%} - dann im Mittel {srows[4]['gain']:.1f} min, höchstens {srows[4]['max_gain']:.0f} min. Dijkstra irrt also **selten**: die Baustelle muss genau dort liegen, wo eine spätere Ankunft die schnellere Route ergibt.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die FIFO-Eigenschaft gilt** | Sinkt die Fahrzeit schneller als die Uhr läuft (endende Baustelle), kann Dijkstra zu spät ankommen; im Stadtnetz mit 40 Baustellen lohnt sich Warten in 12 % der Abfragen, im Mittel 4.0 min. Ohne Warten und ohne FIFO ist die Aufgabe im Allgemeinen schwer (Orda und Rom 1990, Literatur). | Warten erlauben (hier gebaut) |
| **Der Stau ist im Voraus bekannt** | Die Kurven sind hier eine feste Prognose. Ändert sich der Verkehr, müssen Kurven und Routen neu berechnet werden (Literatur: Live-Verkehr, Vorhersagemodelle). | (nicht gebaut) |
| **Es genügt eine Abfrage** | Für viele Abfragen braucht man Vorberechnung wie bei Contraction Hierarchies; mit Funktionen als Kanten wächst der Speicher (Literatur: zeitabhängige Contraction Hierarchies, nicht gebaut). Schon die Ankunftsfunktion am Ziel hat 126 bis 290 Stützstellen (6 × 6 bis 12 × 12). | Contraction Hierarchies |
| **Die Fahrzeit ist die einzige Größe** | Mit einem zweiten Ziel (etwa CO₂) braucht man Labels statt Zahlen; zeitabhängig wird das noch aufwendiger (Literatur). | Mehrkriterien-Demo |
| **Die Kanten sind Straßen, kein Fahrplan** | Bahn und Bus haben feste Abfahrten statt Fahrzeitfunktionen; dafür gibt es eigene Verfahren. | Fahrplan-Routing (RAPTOR, Literatur) |
"""
)
st.caption("Diese Demo ist das neunte Stück der Kürzeste-Wege-Linie: Breitensuche, Dijkstra, bidirektionale Suche, Contraction Hierarchies, Bellman-Ford, Floyd-Warshall, Johnson, Mehrkriterien-Routing und zeitabhängiges Routing.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Gerichteter Graph $G=(V,E)$, Start $s$, Ziel $t$. Jede Kante $e$ hat eine Fahrzeitfunktion $\tau_e:\mathbb{R}\to\mathbb{R}_{\ge0}$, stückweise linear mit Stützstellen alle 30 Minuten (vor Mitternacht und nach 24:00 konstant); hier $\tau_e(t)=b_e\cdot f_{c(e)}(t)$ mit der Grundfahrzeit $b_e$ und dem Faktor $f_{c(e)}\ge 1$ der Klasse.
Die Ankunft bei Abfahrt $t$ ist $a_e(t)=t+\tau_e(t)$; die Ankunft entlang einer Route ist die Verkettung $a_{e_k}\circ\dots\circ a_{e_1}(t_0)$.

**FIFO.** $e$ ist FIFO, wenn $a_e$ nichtfallend ist, also $\tau_e'(t)\ge -1$ überall. Dann ist die Verkettung nichtfallend, und aus $a_u\le a_u'$ folgt, dass die Ankunft an jedem Nachfolger über $u$ nicht später ist als über $u'$.

**Korrektheit des zeitabhängigen Dijkstra.** Bei FIFO und $\tau_e\ge0$ gilt das Optimalitätsprinzip: die früheste Ankunft an $v$ setzt sich aus der frühesten Ankunft an einem Vorgänger $u$ und der Kante $uv$ zusammen. Ein Knoten mit der kleinsten Ankunft in der Warteschlange kann nicht mehr verbessert werden, weil jede andere Route über einen Knoten mit größerer Ankunft führt und danach nicht früher ankommt.
Ohne FIFO gilt das nicht: ein Knoten kann später erreicht werden und dennoch zu einer früheren Ankunft am Ziel führen.

**Warten.** Darf man vor der Kante warten, ist die Ankunft $\tilde a_e(t)=\min_{t'\ge t}\bigl(t'+\tau_e(t')\bigr)$; das Minimum liegt bei $t$ oder an einer Stützstelle nach $t$ (stückweise linear). $\tilde a_e$ ist nichtfallend, also FIFO: das Dijkstra mit $\tilde a_e$ ist exakt.

**Profilsuche.** Für einen Knoten $v$ sei $F_v(d)$ die früheste Ankunft bei Abfahrt $d$ am Start, $F_s(d)=d$. Eine Kante ergibt $F_v'=a_e\circ F_v$ (stückweise linear; neue Stützstellen dort, wo $F_v$ eine Stützstelle von $\tau_e$ überschreitet), zwei Routen ergeben $\min(F,G)$ (neue Stützstellen an den Schnittpunkten). Label-correcting bis sich keine Funktion mehr ändert; bei FIFO terminiert es.

**Reue der statischen Planung.** Für feste Kosten $\bar w_e$ (freie Fahrt: $b_e$; Tagesmittel: Mittelwert von $\tau_e$ über die Stützstellen des Tages) sei $P$ die kürzeste Route. Die Reue bei Abfahrt $t_0$ ist $\bigl(a_P(t_0)-F_t(t_0)\bigr)\big/\bigl(F_t(t_0)-t_0\bigr)$.

**Aufwand.** Zeitabhängiges Dijkstra: wie Dijkstra, $O(m+n\log n)$ mit einer Funktionsauswertung je Kante (Lehrbuchwert; gemessen: dieselben Kantenprüfungen wie statisch). Die Größe der Ankunftsfunktionen ist nicht durch $n$ beschränkt; gemessen: siehe Experimente.

Implementiert in `td_graph.py` (CSR-Graph mit Fahrzeitfunktionen), `td_algorithm.py` (Dijkstra, Warten, Profilsuche, statische Planung und naive Referenzen), `td_scenario.py` (Netze und Stauverläufe), `td_evaluation.py` (Kennzahlen, Tageslauf, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
