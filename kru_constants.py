"""Vorläufig."""
AREA = 100.0                     # Kantenlänge des Gebiets in km
DEPOT_XY = (15.0, 50.0)          # Lage des Depots (Werk) am linken Rand, Filialen zufällig im Gebiet
N_MIN, N_MAX, DEFAULT_N, N_STEP = 5, 200, 30, 1                # Filialen (ohne Depot)
K_MIN, DEFAULT_K = 3, 6                                        # nächste Nachbarn je Knoten; K >= n = vollständig
TERRAIN_MIN, TERRAIN_MAX, DEFAULT_TERRAIN, TERRAIN_STEP = 0.0, 1.0, 0.3, 0.05
SEED_MAX = 999999
DEFAULT_SEED = 35
KINDS = ("depot", "textbook", "chain")
KIND_LABELS = {"depot": "Depot und Filialen (Karte)", "textbook": "Lehrbuchbeispiel (5 Knoten)", "chain": "Ketten-Instanz (Union-Find-Falle)"}
K_OPTIONS = (3, 4, 5, 6, 8, 10, 15, 20, 1000)      # 1000 = vollständiger Graph
TERRAIN_OPTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
UF_MODES = ("naive", "compress", "rank", "full")
SORTS = ("all", "filter")
SORT_LABELS = {"all": "alle Kanten sortieren", "filter": "Filter-Kruskal"}
ROUND_UNIT = 1.0
SWEEP_SEEDS = tuple(range(100000, 100005))
N_SWEEP = (10, 20, 40, 80, 160)
K_SWEEP = (3, 4, 6, 10, 20, 1000)
TERRAIN_SWEEP = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
UF_SIZES = (20, 40, 80, 160)

# --- Gemessene Werte (MEDIAN über 5 feste Instanzen, Seeds 100000-100004; 30 Filialen + Depot, k = 6 nächste Nachbarn, Geländezuschlag 0.3, exakte Kosten;
# --- 2026-09-24, alle Werte über ev.run_config/ev.sweep/ev.uf_experiment nachgerechnet, s. tests/test_claims.py). Kruskal ist deterministisch (kein Zufall im Kern). ---
# ERSPARNIS: der billigste Baum spart gegenüber Einzelleitungen (jede Filiale eine eigene Leitung auf dem kürzesten Weg zum Depot) 73.7 % (n = 30) und wächst mit
#   der Größe: 52.1/67.5/77.6/86.2/90.4 % bei n = 10/20/40/80/160. Der Geländezuschlag ändert die Ersparnis kaum (74.3/73.8/73.7/73.8/73.8 % bei 0/0.2/0.4/0.8/1.0).
# UMWEGE (der Haken): der MST ist billig, nicht kurz. Weg zum Depot im Baum gegen kürzesten Weg: Umweg-Faktor im Median 1.37, im Maximum 3.32 (n = 30); bei n = 160
#   Maximum 7.95. Der Kürzeste-Wege-Baum kostet 47.2 % mehr als der MST (bei k = 3: 24.7 %, bei vollständigem Graph 137.7 %) - dafür hat dort jede Filiale den Faktor 1.
# BETRACHTETE KANTEN: Kruskal hört nach n - 1 angenommenen Kanten auf. Anteil der betrachteten Kanten 62.9 % (k = 6); je dichter der Graph, desto weniger:
#   k = 3/4/6/10/20/vollständig: 88.7/79.0/62.9/42.4/22.2/17.9 %. Filter-Kruskal sortiert dabei nur 78.1/71.6/41.0/27.3/13.5/13.3 % der Kanten.
# UNION-FIND (Zeigerschritte je Stufe, Median über 5 Zufallsinstanzen, vollständiger Graph): n = 20/40/80/160: naiv 151/778/1237/4001, Pfadhalbierung 69/282/438/945, nach Rang
#   105/407/753/1889, Rang + Pfadhalbierung 66/250/385/881. Auf der Ketten-Instanz (n = 20/40/80/160 Knoten): naiv 170/740/3080/12560 (quadratisch, Tiefe n - 1), Pfadhalbierung
#   17/37/77/157, nach Rang 51/111/231/471, beides 51/111/231/471. Pfadhalbierung allein holt fast alles heraus, Rang allein weniger.
# GLEICHSTÄNDE UND STABILITÄT: mit auf ganze km gerundeten Kosten (n = 30, Seed 35) gibt es 4 kostengleiche Bäume, bei n = 100 vollständig gerundet 1536; die Kosten
#   sind unabhängig von der Tie-Reihenfolge, der Baum nicht. Ein zufälliges Rauschen von +-1 % auf allen Kantenkosten ändert den Baum im Median in 30.0 % der Läufe
#   (gerundete Kosten: 80.0 %); bei n = 10/20 in 0 %, bei n = 160 in 70 %.

PRESETS = {
    "Standardfall (Voreinstellung)": {"kind": "depot", "n": 30, "k": 6, "terrain": 0.3, "round_costs": False, "seed": 35, "uf_mode": "full", "sort": "all"},
    "Dünnes Netz (k = 3)": {"kind": "depot", "n": 30, "k": 3, "terrain": 0.3, "round_costs": False, "seed": 35, "uf_mode": "full", "sort": "all"},
    "Vollständiger Graph": {"kind": "depot", "n": 30, "k": 1000, "terrain": 0.3, "round_costs": False, "seed": 35, "uf_mode": "full", "sort": "all"},
    "Viele Filialen (n = 160)": {"kind": "depot", "n": 160, "k": 6, "terrain": 0.3, "round_costs": False, "seed": 35, "uf_mode": "full", "sort": "all"},
    "Gleichstände (gerundet)": {"kind": "depot", "n": 30, "k": 6, "terrain": 0.3, "round_costs": True, "seed": 35, "uf_mode": "full", "sort": "all"},
    "Sehr viele gleich billige Bäume": {"kind": "depot", "n": 100, "k": 1000, "terrain": 0.3, "round_costs": True, "seed": 35, "uf_mode": "full", "sort": "all"},
    "Filter-Kruskal (dicht)": {"kind": "depot", "n": 120, "k": 1000, "terrain": 0.3, "round_costs": False, "seed": 35, "uf_mode": "full", "sort": "filter"},
    "Ketten-Instanz (naives Union-Find)": {"kind": "chain", "n": 60, "k": 6, "terrain": 0.3, "round_costs": False, "seed": 35, "uf_mode": "naive", "sort": "all"},
    "Lehrbuchbeispiel": {"kind": "textbook", "n": 30, "k": 6, "terrain": 0.3, "round_costs": False, "seed": 35, "uf_mode": "full", "sort": "all"},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "30 Filialen, k = 6 Nachbarn, Seed 35: der billigste Baum kostet 466.63 und spart 72.4 % gegenüber Einzelleitungen. Kruskal betrachtet 90 von 113 Kanten. Aber die Wege zum Depot sind im Baum im Median 1.26-mal, im Maximum 3.54-mal so lang wie der kürzeste Weg; der Kürzeste-Wege-Baum kostet 57 % mehr.",
    "Dünnes Netz (k = 3)": "Nur die 3 nächsten Nachbarn als Kandidaten (59 Kanten): Kruskal muss fast alle ansehen (58 von 59). Der Baum ist mit 482.90 3.5 % teurer als bei k = 6, weil einzelne lange Verbindungen fehlen; der Kürzeste-Wege-Baum kostet nur 24.5 % mehr.",
    "Vollständiger Graph": "Alle 465 Paare als Kandidaten: Kruskal betrachtet nur 109 (23 %), der Baum ist derselbe wie bei k = 6 (466.63) - der MST liegt schon in den 6 nächsten Nachbarn. Filter-Kruskal sortiert nur 56 Kanten. Der Kürzeste-Wege-Baum kostet jetzt 123 % mehr.",
    "Viele Filialen (n = 160)": "160 Filialen: der Baum spart 89.6 % gegenüber Einzelleitungen - aber der Umweg-Faktor liegt im Median bei 2.38 und im Maximum bei 9.61. Mehr Kunden teilen sich die Leitung, einzelne Kunden zahlen mit langen Wegen.",
    "Gleichstände (gerundet)": "Kosten auf ganze km gerundet: der Baum kostet 467.00, und es gibt 4 gleich billige Bäume; welcher herauskommt, entscheidet die Tie-Reihenfolge. Ein Rauschen von ±1 % ändert den Baum in 65 % der Versuche.",
    "Sehr viele gleich billige Bäume": "100 Filialen, vollständiger Graph, gerundete Kosten: 1536 verschiedene kostengleiche Bäume (exakt per Matrix-Tree-Satz gezählt). Die Kosten sind immer gleich, der Baum hängt von der Reihenfolge ab.",
    "Filter-Kruskal (dicht)": "120 Filialen, vollständiger Graph (7260 Kanten): Kruskal betrachtet nur 153 (2.1 %). Filter-Kruskal sortiert 161 Kanten statt 7260 und wirft vorher 1085 weg - derselbe Baum.",
    "Ketten-Instanz (naives Union-Find)": "Eine Instanz, an der Union-Find ohne Gegenmaßnahmen quadratisch wird: 60 Knoten, naive Suche 1710 Zeigerschritte (längste Suche 58, Wald so tief wie die Kette), Pfadhalbierung 57, nach Rang 171, beides 171.",
    "Lehrbuchbeispiel": "Fünf Knoten A bis E, sieben Kanten: der MST hat Kosten 15 (B–D 2, C–E 3, A–B 4, B–C 6); die Kante A–D (5) schließt einen Kreis und wird verworfen. Kruskal betrachtet 5 von 7 Kanten.",
}
# Beobachtete Spannweite des MEDIANS der Ersparnis (%) über die 5 festen Instanzen (mit Sicherheitsabstand) für die Karten-Presets mit ihren Einstellungen.
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": (68.0, 79.0),
    "Dünnes Netz (k = 3)": (72.0, 82.0),
    "Vollständiger Graph": (65.0, 76.0),
    "Viele Filialen (n = 160)": (86.0, 94.0),
    "Gleichstände (gerundet)": (68.0, 79.0),
}
