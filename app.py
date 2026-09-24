"""Kruskal – der billigste Weg, alle zu verbinden - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Erstes Stück (Wurzel) der Spannbaum-Reihe der "Konzepte"-Reihe. Ein Depot und n Filialen auf einer Karte sollen mit möglichst wenig Leitung verbunden werden (Fernwärme,
Glasfaser, Sammelleitung). Kruskal sortiert die Kanten und nimmt jede an, die zwei getrennte Gruppen verbindet - der Baum wächst aus vielen kleinen Komponenten zusammen.
Gemessen wird, was dieser billigste Baum wirklich bringt (Ersparnis, Umwege), was Kruskal kostet (Kanten, Union-Find, Sortierung) und wie stabil er ist.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import streamlit as st

import kru_constants as C
from kru_evaluation import SWEEP_LABELS, Settings, analyse, count_msts, instability, sweep, uf_experiment, uf_table
from kru_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from kru_visualization import (
    build_instance,
    build_result,
    build_step_map,
    build_sweep,
    build_uf_bars,
    build_uf_experiment,
)
import kru_algorithm as A

st.set_page_config(page_title="Kruskal – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _uf_experiment(base):
    return uf_experiment(base)


@st.cache_data(show_spinner=False)
def _uf_table(settings):
    return uf_table(analyse(settings).inst)


@st.cache_data(show_spinner=False)
def _ties(settings):
    inst = analyse(settings).inst
    trees, costs = set(), set()
    for tie in range(50):
        r = A.kruskal(inst.n, inst.edges, tie=tie)
        trees.add(tuple(sorted(r.tree)))
        costs.add(round(r.cost, 9))
    return {"distinct_trees": len(trees), "distinct_costs": len(costs), "count": count_msts(inst.n, inst.edges) if inst.n <= 400 else None}


@st.cache_data(show_spinner=False)
def _instability(settings):
    return instability(analyse(settings).inst)


st.title("🌳 Kruskal – der billigste Weg, alle zu verbinden")
st.markdown(
    """
**Erstes Stück der Spannbaum-Reihe** (die Wurzel). Ein Depot und n Filialen liegen auf einer Karte; gesucht ist das Leitungsnetz (Fernwärme, Glasfaser, Sammelleitung), das
**alle** verbindet und dabei möglichst wenig Trasse kostet - ein **minimaler Spannbaum** (MST).

**Kruskal** (1956) sortiert alle möglichen Verbindungen nach Kosten und geht sie der Reihe nach durch: eine Kante wird **angenommen**, wenn sie zwei bisher getrennte Gruppen
verbindet, und **verworfen**, wenn sie nur einen Kreis schließen würde. Die Antwort auf "sind diese beiden schon verbunden?" liefert eine Datenstruktur, **Union-Find**. Der Baum
wächst nicht von einem Punkt aus, sondern aus vielen kleinen Gruppen zusammen. Hier wird gemessen, was dieser billigste Baum bringt, was er verschweigt (Umwege!) und was Kruskal kostet.
"""
)
st.caption(
    "Wurzel der Spannbaum-Reihe; geplante Nachfolger (nicht gebaut): Prim, Borůvka, Euklidischer MST, Gerichteter Spannbaum, Grad-/Hop-beschränkter und Kapazitierter MST, Steiner-Baum, "
    "Prize-Collecting Steiner-Baum, Sensitivität und dynamischer MST, zufällige Spannbäume und Kirchhoff."
)

with st.expander("So funktioniert Kruskal", expanded=True):
    st.markdown(
        """
1. **Sortieren:** alle Kandidatenkanten aufsteigend nach Kosten (Gleichstände nach einer festen Regel).
2. **Der Reihe nach entscheiden:** verbindet die Kante zwei verschiedene Komponenten (Union-Find: verschiedene Wurzeln)? Dann **annehmen** und die beiden Komponenten vereinigen. Sonst **verwerfen**.
3. **Stopp** nach n - 1 angenommenen Kanten - die restlichen Kanten werden nie angesehen.
4. **Warum das optimal ist:** die *Schnitt-Eigenschaft* (die billigste Kante über einen Schnitt gehört zu einem MST) rechtfertigt jede Annahme, die *Kreis-Eigenschaft* (die teuerste Kante eines Kreises gehört zu keinem MST) jede Ablehnung.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.radio("Instanz", options=list(C.KINDS), format_func=lambda v: C.KIND_LABELS[v], key="kind_select",
                    help="Das Lehrbuchbeispiel hat 5 Knoten und 7 Kanten; die Ketten-Instanz ist so gebaut, dass Union-Find ohne Gegenmaßnahmen quadratisch wird (die Kosten sind dort abstrakt, keine Karte).")
    if kind != "textbook":
        n = st.slider("Filialen n" if kind == "depot" else "Knoten n", *bounds("n_slider"), key="n_slider",
                      help="Je mehr Filialen, desto größer die Ersparnis gegenüber Einzelleitungen (Median 52 % bei 10, 90 % bei 160), aber auch desto längere Umwege.")
    else:
        n = C.DEFAULT_N
    if kind == "depot":
        k = st.select_slider("Kandidaten: nächste Nachbarn k", options=list(C.K_OPTIONS), key="k_select", format_func=lambda v: "vollständig" if v == 1000 else str(v),
                             help="Jeder Knoten bietet Kanten zu seinen k nächsten Nachbarn an; \"vollständig\" = alle Paare. Je dichter der Graph, desto weniger Kanten muss Kruskal ansehen (bei vollständigem Graph 18 % statt 89 % bei k = 3).")
        terrain = st.select_slider("Geländezuschlag", options=list(C.TERRAIN_OPTIONS), key="terrain_select", format_func=lambda v: f"{v:g}",
                                   help="Jede Kante kostet Länge mal einen Faktor zwischen 1 und 1 + Zuschlag (Untergrund, Trasse). Die Ersparnis ändert sich dadurch kaum, wohl aber der Baum.")
        rounded = st.radio("Kosten", options=[False, True], format_func=lambda v: "ganzzahlig gerundet (viele Gleichstände)" if v else "exakt", key="round_select",
                           help="Gerundete Kosten erzeugen viele Gleichstände - dann gibt es oft mehrere gleich billige Bäume.")
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        k, terrain, rounded, seed = C.DEFAULT_K, C.DEFAULT_TERRAIN, False, C.DEFAULT_SEED
    uf_mode = st.radio("Union-Find", options=list(C.UF_MODES), format_func=lambda v: {"naive": "naiv", "compress": "Pfadhalbierung", "rank": "Vereinigung nach Rang", "full": "Rang + Pfadhalbierung"}[v],
                       key="uf_select", help="Ändert nur den Aufwand, nie den Baum: alle vier Stufen liefern denselben MST.")
    sort = st.radio("Sortierung", options=list(C.SORTS), format_func=lambda v: C.SORT_LABELS[v], key="sort_select",
                    help="Filter-Kruskal sortiert nur, was es braucht, und wirft Kanten innerhalb einer Komponente vorher weg. Bei paarweise verschiedenen Kosten derselbe Baum.")

sync_query_params({"kind_select": kind, "n_slider": int(n), "k_select": int(k), "terrain_select": float(terrain), "round_select": bool(rounded), "seed_input": int(seed),
                   "uf_select": uf_mode, "sort_select": sort})

settings = Settings(kind, int(n), int(k), float(terrain), bool(rounded), int(seed), uf_mode, sort)
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst, kr = a.inst, a.kruskal
abstract = kind == "chain"

# --- Kruskal in Aktion -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Kruskal in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Kanten entscheiden", 3: "3 · Ergebnis"}
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="kru_step", format_func=lambda s: STEP_LABELS[s])

if step == 1:
    st.markdown(f"**{inst.n - 1 if kind == 'depot' else inst.n} " + ("Filialen" if kind == "depot" else "Knoten") + f"** (Depot ⭐), **{inst.m} Kandidatenkanten**. Gesucht: das billigste Netz, das alle verbindet.")
    st.plotly_chart(build_instance(inst), width="stretch", key="s1_map")
elif step == 2:
    n_steps = len(kr.steps)
    if n_steps >= 1:
        if "kru_edge" in st.session_state:
            st.session_state["kru_edge"] = min(max(1, int(st.session_state["kru_edge"])), n_steps)
        edge_no = st.slider("Betrachtete Kante", 1, max(2, n_steps), key="kru_edge", help="Die Kanten in aufsteigender Reihenfolge der Kosten; Kruskal hört nach der (n - 1)-ten angenommenen Kante auf.") if n_steps > 1 else 1
        edge_no = min(edge_no, n_steps)
        idx, accepted, comps = kr.steps[edge_no - 1]
        u, v, w = inst.edges[idx]
        names = inst.labels if inst.labels is not None else None
        label = f"{names[u]}–{names[v]}" if names else f"{u}–{v}"
        st.markdown(
            f"**Kante {edge_no} von {n_steps} betrachteten** ({label}, Kosten {w:.2f}): "
            + ("**angenommen** - verbindet zwei getrennte Gruppen" if accepted else "**verworfen** - würde einen Kreis schließen")
            + f". Danach {comps} Komponente{'n' if comps != 1 else ''}."
        )
        st.plotly_chart(build_step_map(inst, kr.steps, edge_no), width="stretch", key=f"s2_map_{edge_no}")
        st.caption(f"Insgesamt betrachtet Kruskal {kr.examined} von {inst.m} Kanten ({a.examined_share:.0f} %); die übrigen {inst.m - kr.examined} kommen nie an die Reihe.")
    else:
        st.info("Nichts zu entscheiden.")
else:
    st.plotly_chart(build_result(inst, kr.tree, a.spt_tree, a.stretch_by_node),
                    width="stretch", key="s3_map")
    st.caption("Dick: der billigste Baum (MST). Gestrichelt orange: die Kanten des Kürzeste-Wege-Baums (Dijkstra ab Depot), wo er vom MST abweicht. Punktfarbe: Umweg-Faktor der Filiale.")

st.markdown("---")

# --- Ergebnis ----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was der billigste Baum bringt")
if abstract:
    st.caption("Die Ketten-Instanz hat abstrakte Kosten (keine Karte): Ersparnis und Umwege sind hier ohne Bedeutung, es geht nur um den Union-Find-Aufwand.")
else:
    st.caption(
        "**Ersparnis:** wie viel der Baum gegenüber Einzelleitungen spart (jede Filiale bekommt ihre eigene Leitung auf dem kürzesten Weg zum Depot). **Umweg-Faktor:** Weg im Baum zum Depot geteilt durch den "
        "kürzesten Weg - der billigste Baum ist nicht der Baum der kürzesten Wege. **Betrachtete Kanten:** Anteil, den Kruskal ansehen muss."
    )
m1, m2, m3, m4 = st.columns(4)
m1.metric("Baumkosten (MST)", f"{a.mst_cost:.2f}", delta=f"Prim: {a.prim_cost:.2f}" + (" (gleich)" if abs(a.prim_cost - a.mst_cost) < 1e-6 else " (!)"), delta_color="off")
m2.metric("Ersparnis", "-" if abstract else f"{a.saving:.1f} %", delta="gegen Einzelleitungen", delta_color="off")
m3.metric("Umweg-Faktor", "-" if abstract else f"{a.stretch_median:.2f} / {a.stretch_max:.2f}", delta="Median / Maximum", delta_color="off")
m4.metric("Betrachtete Kanten", f"{kr.examined} / {inst.m}", delta=f"{a.examined_share:.0f} %", delta_color="off")
if not abstract:
    st.caption(f"Der Kürzeste-Wege-Baum hat Kosten {a.spt_cost:.2f}, das sind {a.spt_over_mst:.0f} % mehr als der MST; dafür hat dort jede Filiale den Umweg-Faktor 1.")

st.markdown("---")

# --- Union-Find --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🧱 Was kostet Union-Find?")
st.caption(
    "Zeigerschritte und größte Tiefe je Stufe auf DIESER Instanz (die Auswahl in der Seitenleiste ändert nur, welche Stufe oben gezählt wird). Alle vier liefern denselben Baum; "
    f"gezählt wurde bei der gewählten Stufe: {kr.find_steps} Zeigerschritte, längste Suche {kr.max_find_len}."
)
st.plotly_chart(build_uf_bars(_uf_table(settings)), width="stretch", key="uf_bars")
st.caption(f"Sortierung: {'alle ' + str(inst.m) if sort == 'all' else str(kr.sorted_edges)} Kanten sortiert" + (f", {kr.filtered_out} vorher weggefiltert." if sort == "filter" else "."))

if kind != "textbook":
    if st.button("Union-Find über die Größe messen (kann einen Moment dauern)", key="uf_start"):
        st.session_state["uf_done"] = st.session_state.get("uf_done", set()) | {replace(settings, seed=0)}
    if replace(settings, seed=0) in st.session_state.get("uf_done", set()):
        with st.spinner("Rechne..."):
            rows_uf = _uf_experiment(replace(settings, seed=0))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Zufallsinstanzen** (vollständiger Graph, Median über 5 feste Instanzen)")
            st.plotly_chart(build_uf_experiment(rows_uf, "random"), width="stretch", key="uf_random")
        with c2:
            st.markdown("**Ketten-Instanz** (Union-Find-Falle)")
            st.plotly_chart(build_uf_experiment(rows_uf, "chain"), width="stretch", key="uf_chain")

st.markdown("---")

# --- Gleichstände und Stabilität --------------------------------------------------------------------------------------------------------------

if kind == "depot":
    st.subheader("🎲 Wie eindeutig und wie stabil ist der Baum?")
    st.caption(
        "**Gleichstände:** bei gleichen Kosten entscheidet die Reihenfolge, welcher Baum herauskommt - die Kosten sind immer gleich. **Stabilität:** ein zufälliges Rauschen von ±1 % auf allen Kantenkosten "
        "(20 Versuche) - wie oft ändert sich dadurch der Baum?"
    )
    if st.button("Gleichstände und Stabilität messen", key="tie_start"):
        st.session_state["tie_done"] = st.session_state.get("tie_done", set()) | {settings}
    if settings in st.session_state.get("tie_done", set()):
        ties = _ties(settings)
        t1, t2, t3 = st.columns(3)
        t1.metric("Kostengleiche Bäume (alle)", f"{ties['count']:,}".replace(",", "."), delta="exakt gezählt (Matrix-Tree)", delta_color="off")
        t2.metric("Verschiedene Bäume bei 50 Reihenfolgen", ties["distinct_trees"], delta=f"{ties['distinct_costs']} verschiedene Kosten", delta_color="off")
        t3.metric("Baum ändert sich bei ±1 % Rauschen", f"{_instability(settings):.0f} %", delta="der 20 Versuche", delta_color="off")
    st.markdown("---")

# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------

if kind == "depot":
    st.subheader("📐 Wie hängen Ersparnis, Umwege und Aufwand von den Reglern ab?")
    sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda v: SWEEP_LABELS[v], key="sweep_select")
    metric = st.radio("Kennzahl", options=["saving", "stretch", "examined", "uf", "instability"],
                       format_func=lambda v: {"saving": "Ersparnis / Kürzeste-Wege-Aufschlag", "stretch": "Umweg-Faktor", "examined": "Betrachtete / sortierte Kanten", "uf": "Union-Find-Schritte", "instability": "Instabilität"}[v],
                       key="sweep_metric", horizontal=True)
    base_sweep = replace(settings, seed=0)
    if st.button("Sweep über 5 feste Instanzen berechnen (kann einige Sekunden dauern)", key="sweep_start"):
        st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
    if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
        with st.spinner("Rechne den Sweep über 5 feste Instanzen..."):
            rows_sweep = _sweep(sweep_param, base_sweep)
        label = SWEEP_LABELS[sweep_param]
        if metric == "saving":
            st.plotly_chart(build_sweep(rows_sweep, label, [("saving", "Ersparnis gegen Einzelleitungen (%)", "#2F6B65"), ("spt_over_mst", "Kürzeste-Wege-Baum kostet mehr als MST (%)", "#f58518")], "Prozent"), width="stretch", key="sweep_saving")
        elif metric == "stretch":
            st.plotly_chart(build_sweep(rows_sweep, label, [("stretch_median", "Umweg-Faktor, Median je Instanz", "#4c78a8"), ("stretch_max", "Umweg-Faktor, Maximum je Instanz", "#e45756")], "Umweg-Faktor", ref_line=1.0, ref_label="kürzester Weg"),
                            width="stretch", key="sweep_stretch")
        elif metric == "examined":
            st.plotly_chart(build_sweep(rows_sweep, label, [("examined_share", "betrachtete Kanten (%)", "#2F6B65"), ("sorted_share", "sortierte Kanten mit Filter-Kruskal (%)", "#f58518")], "Anteil aller Kandidatenkanten (%)"), width="stretch", key="sweep_examined")
        elif metric == "uf":
            st.plotly_chart(build_sweep(rows_sweep, label, [("steps_per_edge", "Zeigerschritte je betrachteter Kante", "#7b3fbf"), ("max_find_len", "längste einzelne Suche", "#888888")], "Schritte"), width="stretch", key="sweep_uf")
        else:
            st.plotly_chart(build_sweep(rows_sweep, label, [("instability", "Baum ändert sich bei ±1 % Rauschen (% der Versuche)", "#e45756")], "Prozent"), width="stretch", key="sweep_instability")
        st.caption("Median über 5 feste Instanzen (Seeds 100000–100004), Band = 10. bis 90. Perzentil. Die übrigen Regler stehen wie in der Seitenleiste; Union-Find-Stufe und Sortierung wirken nur auf die Zähler.")
    st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der billigste Baum ist ein gutes Netz** | Er ist billig, aber nicht kurz: die Wege zum Depot sind im Median 1.37-mal, im Maximum 3.3-mal so lang wie der kürzeste Weg; der Kürzeste-Wege-Baum kostet dafür 47 % mehr. Der MST ist eine reine Kostenantwort ohne Rücksicht auf Wegelängen. | Grad-/Hop-beschränkter und Kapazitierter MST (Nachfolger dieser Reihe) |
| **Der Baum ist eindeutig** | Nur bei paarweise verschiedenen Kosten. Bei gerundeten Kosten gibt es oft mehrere gleich billige Bäume; die Tie-Reihenfolge entscheidet, welcher herauskommt. | - |
| **Der Baum ist stabil** | Nein: ein Rauschen von ±1 % auf den Kosten ändert ihn im Median in 30 % der Läufe (mit gerundeten Kosten 80 %). Kleine Preisänderungen können ein ganz anderes Netz bedeuten. | Sensitivität und dynamischer MST (Nachfolger) |
| **Alle Filialen müssen angeschlossen werden, nur Filialen sind Knoten** | Kruskal nimmt nur Kanten zwischen den gegebenen Knoten. Zusätzliche Verzweigungspunkte (Steiner-Punkte) können das Netz kürzen, sind aber NP-schwer. | Steiner-Baum (Nachfolger) |
| **Union-Find braucht Gegenmaßnahmen** | Ohne Kompression und Rang wird es auf der Ketten-Instanz quadratisch. Auf Zufallsinstanzen kostet der naive Wald im Median rund 4x so viele Zeigerschritte wie die volle Variante; Pfadhalbierung allein holt fast alles heraus. | - |
| **Synthetische Instanzen** | Punkte im Quadrat, euklidische Kosten mit Zufallszuschlag, kein Straßennetz, keine Kapazitäten oder Richtungen. Andere Netzstrukturen wurden nicht gemessen. | Echte Trassen (hier nicht gebaut) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gegeben ein zusammenhängender Graph $G=(V,E)$ mit Kosten $w_e$; gesucht ist ein Spannbaum $T \subseteq E$ ($|T| = |V|-1$, azyklisch, zusammenhängend) mit minimalen Kosten $\sum_{e\in T} w_e$.

**Schnitt-Eigenschaft.** Für jeden Schnitt $(S, V\setminus S)$ gehört eine billigste Kante über den Schnitt zu einem MST (bei eindeutig billigster Kante zu jedem).
**Kreis-Eigenschaft.** Eine Kante, die auf einem Kreis strikt teurer ist als alle anderen Kanten dieses Kreises, gehört zu keinem MST.

**Kruskal.** Sortiere $E$ nach $(w_e, \text{Tie-Rang})$; für jede Kante $\{u,v\}$: falls $\text{find}(u) \ne \text{find}(v)$, nimm sie an und vereinige. Laufzeit $O(m \log m)$ für das Sortieren
plus $O(m\,\alpha(m,n))$ für Union-Find mit Rang und Pfadkompression (Tarjan 1975; hier als Pfadhalbierung umgesetzt), $\alpha$ = inverse Ackermann-Funktion.

**Anzahl der MSTs.** Nach Kostenklassen gruppiert ist die Zahl aller MSTs das Produkt, über alle Klassen, der Zahl der Spannbäume im Multigraphen der Klasse-Kanten zwischen den bis dahin
entstandenen Komponenten (Matrix-Tree-Satz: Determinante eines Kofaktors der Laplace-Matrix).

**Umweg-Faktor.** $s(v) = d_T(\text{Depot}, v) / d_G(\text{Depot}, v) \ge 1$, mit $d_T$ = Weglänge im Baum, $d_G$ = kürzester Weg im Kandidatengraphen.

**Literatur.** Kruskal, J. B. (1956). *On the shortest spanning subtree of a graph and the traveling salesman problem.* Proceedings of the AMS 7(1), 48-50. Tarjan, R. E. (1975).
*Efficiency of a good but not linear set union algorithm.* Journal of the ACM 22(2), 215-225. Osipov, V., Sanders, P., & Singler, J. (2009). *The Filter-Kruskal Minimum Spanning Tree Algorithm.* ALENEX 2009.

Implementiert in `kru_algorithm.py` (Kruskal, Filter-Kruskal, Referenzen), `kru_unionfind.py`, `kru_scenario.py` (Instanzen), `kru_evaluation.py` (Kennzahlen, Sweeps, MST-Zählung).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
