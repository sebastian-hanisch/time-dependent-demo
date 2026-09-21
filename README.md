# Zeitabhängiges Routing – die Uhrzeit ändert die Route – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-time-dependent-demo.streamlit.app/)**

Neuntes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Ast von [Dijkstra](../dijkstra-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – das **zeitabhängige Dijkstra** samt Profilsuche – an einem wachsenden Beispiel.
Ein Navigationsgerät rechnet mit festen Fahrzeiten. Im echten Verkehr hängt die Fahrzeit einer Straße von der **Uhrzeit** ab: die schnelle Hauptachse ist um 8 Uhr verstopft, die Landstraße nicht. Beim zeitabhängigen Dijkstra ist das Label eines Knotens seine **Ankunftszeit**,
und jede Kante wird zu der Zeit ausgewertet, zu der man an ihrem Anfang ankommt. Der Aufwand bleibt derselbe – die Frage ist, **wann das noch stimmt**: genau dann, wenn niemand durch **späteres Losfahren früher ankommt** (FIFO-Eigenschaft). Eine **Profilsuche** liefert die Ankunftszeit
als Funktion der Abfahrtszeit für den ganzen Tag. Wer mit festen Kosten plant, verliert Zeit; endet eine Baustelle, kann Dijkstra sogar falsch liegen, und Warten lohnt sich.

**Einordnung in die Reihe (die Kanten des Graphen):**
```
bfs-demo (Wurzel: Kanten zählen, nicht Kosten)                                       [gebaut]
  └─ dijkstra-demo (Kosten korrekt, blind in alle Richtungen)                        [gebaut]
       ├─ bidirectional-demo → contraction-hierarchies-demo                          [gebaut]
       ├─ bellman-ford-demo ─┐                                                       [gebaut]
       │   floyd-warshall-demo ─┴→ johnson-demo (Konvergenz: Umgewichtung)           [gebaut]
       ├─ multicriteria-demo (Zeit gegen CO₂, Pareto)                                [gebaut]
       └─ time-dependent-demo (Kosten hängen von der Uhrzeit ab)                     [dieses Stück]
```

## Quellen

| Bestandteil | Quelle |
|---|---|
| Verfahren (zeitabhängiges Dijkstra, FIFO-Bedingung, Warten) | Cooke und Halsey (1966), Dreyfus (1969), Orda und Rom (1990); die Bücher (*Grokking Algorithms*, *Optimization Algorithms*) behandeln keine zeitabhängigen Kosten, es gibt kein Buchbeispiel zu spiegeln |
| Umsetzung, Profilsuche (Label-correcting mit stückweise linearen Funktionen), Wartehülle, statische Planung, Tageslauf | eigen |
| Alle Netze und Stauverläufe | **eigene Graphen und Erzeuger**: kleines Netz mit drei Tagesrouten, Baustelle (Überholen), Stadtnetz mit Hauptachsen, Zufallsnetz; die Staukurve (zwei Spitzen) und alle Fahrzeiten sind erfunden |
| Zahlen | **eigene Messungen** an diesen Netzen |

Aus den Büchern stammt keine Zahl, kein Graph und kein Text. **Keine Verkehrsdaten und kein OpenStreetMap-Auszug**, also keine ODbL-Pflichten.

## Ergebnis (Zahlen aus den Tests)

| Frage | Ergebnis |
|---|---|
| Kleines Netz (7 Orte, Stau-Stärke 1.5, Abfahrt 8:00) | ✅ die beste Route wechselt im Lauf des Tages **achtmal** zwischen **drei** Routen (Ring nachts, Markt in den Übergängen, Dorf in der Spitze). Um 8:00 braucht die beste Route **18.3 min**; wer mit den Tagesmittel-Kosten plant (Route über den Markt, nie lange die beste), braucht **25.5 min (+39 %)**, mit freier Fahrt geplant 33.4 min |
| Stadtnetz (8 × 8, Seed 10) | ✅ fünf verschiedene beste Routen im Lauf des Tages; um 8:00 **19.9 min** gegen **21.7 min (+8.8 %)** für die mit dem Tagesmittel geplante Route, nachts beide 9.4 min |
| Mehrzeit der statischen Planung gegen die Stau-Stärke (Stadtnetz 8 × 8, Mittel über 5 Netze × 20 Paare) | ⚠️ um 8:00 bei Stärke 0.5 / 1 / 1.5 / 2 / 3: **0.4 / 2.2 / 5.9 / 10.3 / 17.8 %** (Höchstwert bei 1.5: 28 %; mit freier Fahrt geplant im Mittel 8.4 %, Höchstwert 37 %); nachts und mittags nur 0.5 % bei Stärke 1.5 – die Mehrzeit entsteht in der Spitze. Die beste Route um 8:00 ist in 34 / 66 / 96 / 100 / 100 % der Paare eine andere als um 3:00; im Zufallsnetz (60 Knoten) 38 % der Paare und 1.8 % Mehrzeit |
| Aufwand | ✅ das zeitabhängige Dijkstra prüft **genau dieselben Kanten** wie das statische (120 / 224 / 360 / 528 bei 36 / 64 / 100 / 144 Knoten): die Zeitabhängigkeit kostet keine zusätzlichen Schritte |
| Profilsuche | ⚠️ exakt (stimmt an jeder Abfahrtszeit mit dem Dijkstra überein), aber teuer: **21 409 bis 263 790 Stützstellen** verarbeitet gegen 11 520 bis 50 688 Kantenprüfungen für 96 einzelne Läufe (alle 15 min); die Ankunftsfunktion am Ziel hat 126 bis 290 Stützstellen (6 × 6 bis 12 × 12). Ihr Vorteil ist nicht der Aufwand, sondern dass keine Abfahrtszeit durch das Raster fällt |
| Baustelle (Überholen, FIFO verletzt) | ❌ das Dijkstra ohne Warten kommt um 8:00 nach **58.9 min** an, mit erlaubtem Warten nach **50.0 min** (8.9 min weniger); auch die beste Route ohne Warten (Brute-Force) ist noch später als mit Warten |
| Wann irrt Dijkstra im Stadtnetz? | ⚠️ **selten**: ohne Baustellen nie; bei 10 / 20 / 40 Baustellen (8 × 8) lohnt sich Warten in **1.0 / 3.3 / 11.7 %** der Abfragen, bei 40 im Mittel 4.0 min (höchstens 15 min) |
| Korrektheit | ✅ zeitabhängiges Dijkstra = **Brute-Force über alle Routen** (FIFO-Netze) bzw. mit Warten (auch ohne FIFO); ohne Stau = gewöhnliches Dijkstra (networkx); Wartehülle ändert bei FIFO nichts; die Profilsuche stimmt an jeder Stichprobe mit dem Dijkstra überein; statische Route = kürzeste Route für die festen Kosten (networkx); Reue nie negativ |

Die Zähler (Kantenprüfungen, Stützstellen) sind Schritte des Verfahrens und plattformfest. Laufzeiten stehen in der App nur als Messwerte (reines Python) und werden nirgends behauptet oder getestet.

## Was die Demo zeigt

1. **Zeitabhängiges Routing in Aktion** (Regler über die festgelegten Knoten + Abspielen): die Karte mit den Knoten nach Minuten seit der Abfahrt, die beste Route, die mit dem Tagesmittel geplante Route (gestrichelt), bei Baustellen die beste Route mit Warten; daneben **Fahrzeit gegen Abfahrtszeit** für den ganzen Tag (beste Route, Tagesmittel-Plan, Plan mit freier Fahrt, bei Baustellen mit Warten); beim kleinen Netz der **Tageslauf der besten Route** als Tabelle. Der Regler **Abfahrtszeit** (15-Minuten-Schritte) gilt für Karte und Kennzahlen.
2. **Die Uhrzeit ändert die Route:** beste Route, mit Tagesmittel geplante Route und ihre Reue, wie oft die beste Route im Tagesverlauf wechselt, Kantenprüfungen zeitabhängig gegen statisch; das Urteil unterscheidet Stau (Route ändert sich), gleiche Route zu dieser Uhrzeit, kein Stau (Stärke 0) und verletzte FIFO-Eigenschaft (Warten spart X Minuten oder bringt zu dieser Zeit nichts).
3. **Vergleich** (Expander: statisches, zeitabhängiges Dijkstra, 96 Läufe, Profilsuche); **Experimente auf Knopfdruck**: Mehrzeit gegen Stau-Stärke und Uhrzeit; Aufwand gegen Netzgröße (Dijkstra gegen Profilsuche); Fehlerquote gegen Baustellen.
4. **Wo die Annahmen enden** (Tabelle; FIFO, bekannter Stau, viele Abfragen und zeitabhängige Contraction Hierarchies, mehrere Ziele, Fahrpläne) und **Mathematische Formulierung** (Fahrzeitfunktion, FIFO, Korrektheit, Wartehülle, Profilsuche, Reue, Aufwand als Lehrbuchwert gekennzeichnet).

Bedienung: Beispielnetz per Schnellstart-Knopf laden oder in der Seitenleiste Netz, Stau-Stärke und Abfahrtszeit wählen; beim Stadtnetz Größe, Baustellen und Seed, beim Zufallsnetz Knoten, Grad und Seed. Die Adresszeile spiegelt die Konfiguration (Permalink). Höchstens 144 Knoten im Stadtnetz, 120 im Zufallsnetz.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `td_graph.py` | Graph in CSR-Form mit Fahrzeitfunktionen (Stützstellen alle 30 min), Auswertung, Wartehülle |
| `td_algorithm.py` | zeitabhängiges Dijkstra (mit/ohne Warten), Profilsuche, statische Planung, Reue, Brute-Force-Referenz |
| `td_scenario.py` | Netze: kleines Netz, Baustelle, Stadtnetz mit Hauptachsen, Zufallsnetz; Staukurve und Baustellenverlauf |
| `td_evaluation.py` | Kennzahlen, Tageslauf, Bildfolge, Experimente |
| `td_visualization.py`, `td_presets.py`, `td_constants.py` | Abbildungen, Presets und Permalink, Konstanten |

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/`. Jede Zahl in Hilfetexten, Presets und Tabellen ist in `tests/test_claims.py` belegt; die Kreuzprobe läuft gegen eine Brute-Force-Suche über alle Routen und networkx (Netze mit Parallelkanten, Nullkosten, unerreichbaren Zielen und identischem Start und Ziel); ein Regressionstest klickt "▶️ Abspielen" auf Netzen mit mehreren Bildern.
