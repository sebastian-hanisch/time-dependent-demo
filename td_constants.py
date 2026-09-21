"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

SPACING = 500.0                    # Meter zwischen benachbarten Kreuzungen im erzeugten Stadtnetz
JITTER = 0.2                       # Lageabweichung der Kreuzungen in Blocklängen
AXIS_EVERY = 4                     # jede vierte Zeile und Spalte ist Hauptachse
SPEED_LOCAL, SPEED_ARTERIAL = 30.0, 60.0            # km/h bei freier Fahrt
AMP_LOCAL, AMP_ARTERIAL = 0.3, 1.2                  # wie stark der Stau die Klasse trifft (Faktor 1 + Stärke * Anteil * Staukurve)
SITE_DELAY = 45.0                  # Zuschlag einer Baustelle im Stadtnetz in Minuten (sinkt zwischen 8:00 und 8:30 auf 0)
NOISE = 0.25                       # zufällige Streuung der Grundfahrzeit je Kante (Ampeln, Belag)

# Staukurve: Anteil des Höchststaus je halbe Stunde von 0:00 bis 24:00 (zwei Spitzen, morgens und abends)
RUSH = {6.5: 0.3, 7.0: 0.7, 7.5: 1.0, 8.0: 1.0, 8.5: 0.9, 9.0: 0.6, 9.5: 0.3, 10.0: 0.15, 15.5: 0.15, 16.0: 0.5, 16.5: 0.9, 17.0: 1.0, 17.5: 1.0, 18.0: 0.8, 18.5: 0.5, 19.0: 0.25, 19.5: 0.1}

NETS = ("small", "site", "city", "random")
NET_LABELS = {
    "small": "🔀 Kleines Netz (die Route wechselt mit der Uhrzeit)",
    "site": "🚧 Baustelle (Überholen: Warten lohnt)",
    "city": "🏙️ Stadtnetz mit Hauptachsen (erzeugt)",
    "random": "🕸️ Zufallsnetz (erzeugt)",
}
SMALL_NETS = ("small", "site")                             # eigener Graph mit Namen, feste Aufgabe
FIXED_NETS = ("small", "site")

STRENGTH_MIN, STRENGTH_MAX, DEFAULT_STRENGTH = 0.0, 3.0, 1.5     # Stau-Stärke: 0 = kein Stau
SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 4, 12, 8                # n = Seite²
SITES_MIN, SITES_MAX, DEFAULT_SITES = 0, 40, 0             # Baustellen im Stadtnetz (Kanten, die überholt werden können)
NODES_MIN, NODES_MAX, DEFAULT_NODES = 20, 120, 60
DEGREE_MIN, DEGREE_MAX, DEFAULT_DEGREE = 2.0, 5.0, 3.0
DEPART_MIN, DEPART_MAX, DEFAULT_DEPART = 0, 1440, 480      # Abfahrtszeit in Minuten seit Mitternacht (Schritt 15)
DEFAULT_SEED = 7
DEFAULT_NET = "small"

SWEEP_SEEDS = tuple(range(100000, 100005))

COLORS = {"td": "#1f77b4", "static": "#d62728", "wait": "#2ca02c", "start": "#111111", "goal": "#ff7f0e", "arterial": "#ff7f0e"}

DEPART_OPTIONS = tuple(range(0, 1440, 15))                # Abfahrtszeiten des Reglers (Minuten seit Mitternacht)


def hhmm(minutes):
    """Uhrzeit "8:05" aus Minuten seit Mitternacht (über 24:00 hinaus: am Folgetag)."""
    m = int(round(minutes))
    return f"{(m // 60) % 24}:{m % 60:02d}"


_BASE = dict(strength=DEFAULT_STRENGTH, side=DEFAULT_SIDE, sites=DEFAULT_SITES, nodes=DEFAULT_NODES, degree=DEFAULT_DEGREE, depart=DEFAULT_DEPART, seed=DEFAULT_SEED)
PRESETS = {
    "🔀 Kleines Netz": {**_BASE, "net": "small"},
    "🚧 Baustelle": {**_BASE, "net": "site"},
    "🏙️ Stadtnetz": {**_BASE, "net": "city", "seed": 10},
    "🕸️ Zufallsnetz": {**_BASE, "net": "random", "seed": 13},
}
PRESET_HELP = {
    "🔀 Kleines Netz": "Kleines Netz (7 Orte, Stau-Stärke 1.5, Abfahrt 8:00): im Lauf des Tages wechselt die beste Route achtmal, zwischen drei verschiedenen - über den Ring bei freier Fahrt, über die Landstraße durch das Dorf in der Spitze, über den Markt in den Übergängen. Um 8:00 braucht die beste Route 18.3 min; wer mit den Tagesmittel-Kosten plant (Route über den Markt), braucht 25.5 min, also 39 % länger.",
    "🚧 Baustelle": "Baustelle (6 Orte, Abfahrt 8:00): die Baustelle auf der letzten Strecke endet um 8:30 - wer später hinfährt, kommt früher an, die FIFO-Eigenschaft ist verletzt. Dijkstra ohne Warten kommt um 8:00 nach 58.9 min an, mit erlaubtem Warten nach 50.0 min (8.9 min weniger).",
    "🏙️ Stadtnetz": "Stadtnetz (8 × 8, Stau-Stärke 1.5, Seed 10, Abfahrt 8:00): fünf verschiedene beste Routen im Lauf des Tages. Um 8:00 braucht die beste Route 19.9 min, die mit dem Tagesmittel geplante 21.7 min (+8.8 %); nachts sind beide gleich (9.4 min).",
    "🕸️ Zufallsnetz": "Zufallsnetz (60 Knoten, Grad 3, Stau-Stärke 1.5, Seed 13, Abfahrt 8:00): um 8:00 braucht die beste Route 27.1 min, die mit dem Tagesmittel geplante 32.8 min (+21.3 %).",
}
