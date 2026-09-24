# Kruskal – der billigste Weg, alle zu verbinden – Streamlit-Demo

Erstes Stück (Wurzel) der **Spannbaum-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning". Ein Depot und n Filialen liegen auf einer Karte; gesucht ist das Leitungsnetz (Fernwärme, Glasfaser, Sammelleitung), das **alle** verbindet und dabei möglichst wenig Trasse kostet: ein **minimaler Spannbaum** (MST). **Kruskal** (1956) sortiert alle möglichen Verbindungen nach Kosten und geht sie der Reihe nach durch: eine Kante wird **angenommen**, wenn sie zwei bisher getrennte Gruppen verbindet, und **verworfen**, wenn sie einen Kreis schließen würde. Die Frage "sind diese beiden schon verbunden?" beantwortet **Union-Find**. Der Baum wächst nicht von einem Punkt aus, sondern aus vielen kleinen Komponenten zusammen.

**Einordnung in die Reihe:** im Portfolio gab es zu Spannbäumen bisher nichts (Kanten hierher: Single-Linkage-Clustering in [agglomerative-demo](../agglomerative-demo) ist ein MST auf Distanzen; die Basislösung des Netzwerksimplex in [network-flow-demo](../network-flow-demo) ist ein Spannbaum; der Kürzeste-Wege-Baum aus der Kürzeste-Wege-Linie ist der Kontrast). Geplant sind elf Stücke, dies ist die Wurzel:

```
Kruskal (Wurzel)                                                                           [DIESES STÜCK]
 ├─ Prim (Kontrast: wächst von einem Punkt)                                                [nicht gebaut]
 ├─ Borůvka (Kontrast: alle Komponenten parallel)                                          [nicht gebaut]
 ├─ Euklidischer MST (keine n²-Kantenliste, Delaunay)                                      [nicht gebaut]
 ├─ Gerichteter Spannbaum (Chu-Liu/Edmonds)                                                [nicht gebaut]
 ├─ Bottleneck-/Grad-/Hop-beschränkter Spannbaum → Kapazitierter MST                       [nicht gebaut]
 ├─ Steiner-Baum → Prize-Collecting Steiner-Baum                                           [nicht gebaut]
 ├─ MST-Sensitivität & dynamischer MST                                                     [nicht gebaut]
 └─ Zufällige Spannbäume & Kirchhoff                                                       [nicht gebaut]
```

Ergebnis in Kürze: **Der billigste Baum spart viel, aber er ist nicht kurz, und er ist nicht stabil.** Gegenüber getrennten Einzelleitungen spart er bei 30 Filialen **73,7 %** (52 % bei 10, 90 % bei 160 Filialen). Der Preis: die Wege zum Depot sind im Baum im Median **1,37-mal**, im Maximum **3,32-mal** so lang wie der kürzeste Weg (bei 160 Filialen bis **7,95-mal**); der Kürzeste-Wege-Baum kostet dafür 47 % mehr. Ein Rauschen von ±1 % auf allen Kantenkosten ändert den Baum im Median in **30 %** der Läufe (bei auf ganze km gerundeten Kosten 80 %). Kruskal selbst muss nur einen Teil der Kanten ansehen (63 % bei k = 6 Nachbarn, **18 %** beim vollständigen Graphen), und beim Union-Find holt die **Pfadhalbierung allein** fast alles heraus - die Vereinigung nach Rang allein deutlich weniger.

| Frage | Ergebnis (30 Filialen + Depot, k = 6 nächste Nachbarn, Geländezuschlag 0,3, exakte Kosten, sofern nicht anders angegeben; **Median** über 5 feste Instanzen, Seeds 100000–100004; vollständig deterministisch) |
|---|---|
| **Ist Kruskal optimal?** | ✅ ja, direkt geprüft: Schnitt- und Kreis-Eigenschaft für jede Kante, Brute-Force-Aufzählung aller Spannbäume (kleine Graphen mit 5 Knoten), scipy und ein unabhängiges Prim (Kosten immer gleich, bei paarweise verschiedenen Kosten derselbe Baum) |
| **Wie viel spart der Baum?** | **73,7 %** gegen Einzelleitungen; **52,1/67,5/77,6/86,2/90,4 %** bei n = 10/20/40/80/160; der Geländezuschlag ändert es kaum (74,3/73,8/73,7/73,8/73,8/73,8 % bei 0/0,2/0,4/0,6/0,8/1,0) |
| **Ist der Baum auch kurz?** | ⚠️ nein: Umweg-Faktor (Weg im Baum / kürzester Weg) **1,37 im Median, 3,32 im Maximum**; bei n = 160 Maximum 7,95. Der Kürzeste-Wege-Baum kostet **47,2 %** mehr (k = 3: 24,7 %, vollständiger Graph: 137,7 %) |
| **Wie viele Kanten muss Kruskal ansehen?** | Anteil betrachteter Kanten bei k = 3/4/6/10/20/vollständig: **88,7/79,0/62,9/42,4/22,2/17,9 %** - je dichter der Graph, desto weniger. Filter-Kruskal sortiert dabei nur 78,1/71,6/41,0/27,3/13,5/13,3 % |
| **Was bringt Union-Find?** | ⚠️ Zeigerschritte auf Zufallsinstanzen (vollständiger Graph) bei n = 20/40/80/160: naiv **151/778/1237/4001**, Pfadhalbierung 69/282/438/945, nach Rang 105/407/753/1889, Rang + Pfadhalbierung **66/250/385/881**. Auf der Ketten-Instanz (n = 20/40/80/160): naiv **170/740/3080/12560** (quadratisch, Tiefe n − 1), Pfadhalbierung 17/37/77/157, Rang 51/111/231/471 |
| **Ist der Baum eindeutig?** | nur bei paarweise verschiedenen Kosten: mit gerundeten Kosten gibt es bei n = 30 (Seed 35) **4** kostengleiche Bäume, bei n = 100 vollständig gerundet **1536** (exakt per Matrix-Tree-Satz gezählt); die Kosten hängen nie von der Tie-Reihenfolge ab |
| **Wie stabil ist der Baum?** | ⚠️ ±1 % Rauschen ändert ihn in **30 %** der Läufe (n = 10/20/40/80/160: 0/0/30/60/70 %; gerundete Kosten: 80 %) |

## Was die Demo zeigt

1. **Kruskal in Aktion** (Schritt-Slider): **Instanz** → **Kanten entscheiden** (Slider über die sortierten Kanten: dicke Linie = angenommen, gestrichelt rot = verworfen, orange = aktuelle Kante, Punktfarbe = Komponente; darüber die Begründung "verbindet zwei getrennte Gruppen" / "schließt einen Kreis") → **Ergebnis** (billigster Baum und Kürzeste-Wege-Baum überlagert, Punktfarbe = Umweg-Faktor der Filiale).
2. **Was der billigste Baum bringt:** Baumkosten (mit der Prim-Kontrollrechnung daneben), Ersparnis, Umweg-Faktor, betrachtete Kanten.
3. **🧱 Union-Find:** Zeigerschritte und Tiefe je Stufe auf dieser Instanz; auf Abruf der Verlauf über die Größe für Zufallsinstanzen und die Ketten-Instanz.
4. **🎲 Gleichstände und Stabilität** (auf Abruf): Anzahl aller kostengleichen Bäume, verschiedene Bäume bei 50 Tie-Reihenfolgen, Anteil der Rausch-Versuche, die den Baum ändern.
5. **📐 Sweeps** über n, k und Geländezuschlag (Ersparnis / Kürzeste-Wege-Aufschlag, Umweg-Faktor, betrachtete und sortierte Kanten, Union-Find-Schritte, Instabilität; 5 feste Instanzen, Median, 10.–90. Perzentil-Band).
6. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an".

Regler: Instanz (Depot und Filialen / **Lehrbuchbeispiel** / **Ketten-Instanz**), Filialen n (5–200), Kandidaten k (3 bis vollständig), Geländezuschlag (0–1), Kosten (exakt / gerundet), Seed (+ 🎲), **Union-Find-Stufe** (naiv / Pfadhalbierung / Rang / beides) und **Sortierung** (alle / Filter-Kruskal) - die beiden letzten ändern den Aufwand, nie den Baum (als Test hinterlegt). Kein Zufall im Kern.

## Messwerte der Presets

| Preset | Instanz | Ergebnis |
|---|---|---|
| Standardfall (Voreinstellung) | 30 Filialen, k = 6, Seed 35 | Kosten 466,63, Ersparnis 72,4 %, 90 von 113 Kanten betrachtet, Umweg-Faktor 1,26 / 3,54, Kürzeste-Wege-Baum +57 %, Instabilität 35 % |
| Dünnes Netz (k = 3) | 59 Kanten | 58 von 59 betrachtet; Kosten 482,90 (+3,5 % gegen k = 6); Kürzeste-Wege-Baum +24,5 % |
| Vollständiger Graph | 465 Kanten | 109 betrachtet (23 %), derselbe Baum wie bei k = 6, Filter-Kruskal sortiert 56 Kanten, Kürzeste-Wege-Baum +123 % |
| Viele Filialen (n = 160) | Seed 35 | Ersparnis 89,6 %, Umweg-Faktor 2,38 / 9,61 |
| Gleichstände (gerundet) | Seed 35 | Kosten 467,00, 4 kostengleiche Bäume, Instabilität 65 % |
| Sehr viele gleich billige Bäume | n = 100, vollständig, gerundet | 1536 kostengleiche Bäume |
| Filter-Kruskal (dicht) | n = 120, vollständig (7260 Kanten) | 153 betrachtet (2,1 %); 161 Kanten sortiert statt 7260, 1085 vorher weggefiltert |
| Ketten-Instanz (naives Union-Find) | 60 Knoten | naiv 1710 Zeigerschritte (längste Suche 58) gegen 57 (Pfadhalbierung), 171 (Rang, beides) |
| Lehrbuchbeispiel | 5 Knoten, 7 Kanten | Kosten 15, A–D (5) schließt einen Kreis und wird verworfen, 5 von 7 Kanten betrachtet |

Die einzelne Instanz weicht von den Medianen ab - die Mediane sind die belastbaren Zahlen; die Karten-Presets prüfen sich zusätzlich über die 5 festen Instanzen gegen eine gemessene Spannweite des Medians der Ersparnis, die Aussagen der Presets sind als eigene Tests hinterlegt (`tests/test_presets.py`).

## Modell und Verfahren

- **Instanz** (`kru_scenario.py`): Depot (Knoten 0, am linken Rand) und n Filialen im Quadrat; Kantenkosten = euklidischer Abstand mal ein Geländefaktor in [1, 1 + Zuschlag], der nur vom Knotenpaar und Seed abhängt - wer k ändert, streicht Kanten, ändert aber keine Kosten. Kandidatengraph vollständig oder die k nächsten Nachbarn (Zusammenhang garantiert). "Kosten runden" erzeugt bewusst viele Gleichstände. Zwei Fixtures: ein Lehrbuchbeispiel und eine Ketten-Instanz für die Union-Find-Falle.
- **Union-Find** (`kru_unionfind.py`): vier Stufen (naiv, Pfadhalbierung, Vereinigung nach Rang, beides) mit Zählern für Zeigerschritte, längste Suche und Tiefe.
- **Kruskal** (`kru_algorithm.py`): Sortieren nach (Kosten, Tie-Rang), Annahme/Verwerfen per Union-Find, Stopp nach n − 1 Kanten; **Filter-Kruskal** (Osipov, Sanders & Singler 2009): quicksort-artiges Teilen am Median dreier Kanten, die leichte Hälfte zuerst, aus der schweren vor dem Sortieren alle Kanten innerhalb einer Komponente entfernt. Die Tie-Reihenfolge ist wählbar (lexikographisch, umgekehrt, Seed). Referenzen: unabhängiges Prim, Dijkstra-Baum, Baumdistanzen.
- **Auswertung** (`kru_evaluation.py`): Ersparnis gegen Einzelanbindung, Umweg-Faktor, Kürzeste-Wege-Aufschlag, betrachtete und sortierte Kanten, Zeigerschritte, Instabilität (±1 % Rauschen, 20 Versuche) und die **exakte Anzahl aller MSTs** (Produkt über die Kostenklassen der Zahl der Spannbäume im Klassen-Multigraphen, Determinante per Bareiss-Verfahren).

## Was nicht funktioniert hat / Grenzen

- **Der billigste Baum ist ein gutes Netz - nur bedingt.** Er ist eine reine Kostenantwort: Wege zum Depot bis zum 3,3-Fachen (bei 160 Filialen 8-Fachen) des kürzesten Weges. Wer Umwege begrenzen will, braucht Grad-, Hop- oder Kapazitätsgrenzen (geplante Nachfolger).
- **Der Baum ist nicht stabil.** ±1 % Rauschen ändert ihn in 30 % der Läufe; bei kleinen Instanzen (10, 20 Filialen) in keinem, bei 160 in 70 %. Die Sensitivität wird in einem eigenen Stück untersucht.
- **Erwartung "naives Union-Find ist auf Zufallsinstanzen kaum schlechter" - widerlegt in der Größenordnung:** auf Zufallsinstanzen kostet der naive Wald bei n = 160 rund 4,5-mal so viele Zeigerschritte wie die volle Variante (4001 gegen 881), auf der Ketten-Instanz 27-mal (12560 gegen 471). Aber: die Pfadhalbierung allein (945) kommt nahe an die volle Variante (881), die Vereinigung nach Rang allein (1889) nicht.
- **Geländezuschlag:** ändert die Ersparnis praktisch nicht (Spanne 0,6 Prozentpunkte über den ganzen Bereich); er verschiebt nur, welcher Baum entsteht.
- **Nicht gebaut:** Prim, Borůvka, Euklidischer MST, gerichtete Spannbäume, Nebenbedingungen, Steiner-Bäume, Sensitivität und dynamischer MST, Kirchhoff. Die asymptotisch besseren Verfahren (Karger-Klein-Tarjan 1995, erwartet linear; Chazelle 2000; Pettie-Ramachandran 2002, optimal) werden nicht gebaut: in der Praxis gewinnen (Filter-)Kruskal, Prim und Borůvka-Varianten auf GPUs.
- **Synthetische Instanzen:** Punkte im Quadrat, euklidische Kosten mit Zufallszuschlag, kein Straßennetz, keine Kapazitäten oder Richtungen; die Ketten-Instanz hat abstrakte Kosten (Ersparnis und Umwege sind dort ohne Bedeutung und ausgeblendet).

## Verifikation

- **Spannbaum:** n − 1 Kanten, zusammenhängend, azyklisch, Kosten gegen unabhängige Neuberechnung; **Optimalität direkt** über Schnitt- und Kreis-Eigenschaft (jede Baum- und Nichtbaumkante), gegen Brute-Force-Aufzählung, scipy und ein unabhängiges Prim.
- **Gleichstände:** verschiedene Tie-Reihenfolgen liefern gleiche Kosten und gültige Bäume; vollständiger Graph mit gleichen Kosten erreicht viele der 125 Bäume; die MST-Zählung stimmt mit der Aufzählung überein (Cayley, Kreis, Lehrbuch, gerundete Zufallsgraphen); die Bareiss-Determinante stimmt mit numpy überein.
- **Union-Find:** alle vier Stufen liefern denselben Baum und dieselben Schritte; Modell gegen naive Komponentenliste; Rang-Tiefe ≤ ⌈log₂ n⌉, naive Tiefe n − 1 auf der Kette; Zähler gegen Neuzählung.
- **Buchführung:** jede Kante höchstens einmal betrachtet, Komponentenzahl nach jedem Schritt = n − Angenommene, Stopp sofort nach n − 1, Kosten in den Schritten nicht fallend; **Filter-Kruskal** liefert bei paarweise verschiedenen Kosten denselben Baum, sortiert nie mehr Kanten als m.
- **Sonderfälle:** ein und zwei Knoten, Baum als Eingabe, unzusammenhängender Graph (Wald, ausdrücklich markiert), Dijkstra-Baum gegen Floyd-Warshall.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt**, über dieselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`/`ev.uf_experiment`), NIE über ein Ad-hoc-Skript; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt, jede Kante, alle Instanz-Typen, Extremwerte, Würfel, Permalink-Grenzen, Instanzwechsel, Union-Find-Experiment, Gleichstände und Sweeps auf Abruf, Footer). 405 Tests.

Literatur: Kruskal, J. B. (1956). *On the shortest spanning subtree of a graph and the traveling salesman problem.* Proceedings of the American Mathematical Society 7(1), 48-50. Tarjan, R. E. (1975). *Efficiency of a good but not linear set union algorithm.* Journal of the ACM 22(2), 215-225. Osipov, V., Sanders, P., & Singler, J. (2009). *The Filter-Kruskal Minimum Spanning Tree Algorithm.* ALENEX 2009, 52-61.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Instanz-Umschalter, Schritte (mit Kanten-Slider), Ergebnis, Union-Find, Gleichstände, Sweeps, Grenzen, Mathe |
| `kru_algorithm.py`, `kru_unionfind.py` | Kruskal, Filter-Kruskal, Referenzen (Prim, Dijkstra-Baum, Baumdistanzen); Union-Find in vier Stufen |
| `kru_scenario.py` | Depot-Instanz, Lehrbuchbeispiel, Ketten-Instanz |
| `kru_constants.py` | Konstanten, Presets, gemessene Werte |
| `kru_evaluation.py` | Kennzahlen, Sweeps, Union-Find-Experiment, Instabilität, MST-Zählung |
| `kru_presets.py`, `kru_visualization.py` | Permalink/Presets, Plotly-Figuren (Schrittwiedergabe, Ergebnis-Karte, Union-Find, Sweeps) |
| `tests/` | Zentrale Korrektheitskette (Schnitt-/Kreis-Eigenschaft, Brute-Force, scipy), Szenario/Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
