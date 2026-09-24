"""Auswertung: was leistet der billigste Spannbaum, und was kostet Kruskal? Ein Kruskal-Lauf (Union-Find-Stufe, Sortierung) gegen unabhängiges Prim (Kontrollrechnung),
die Einzelanbindung (jede Filiale bekommt eine eigene Leitung auf dem kürzesten Weg zum Depot) und den Kürzeste-Wege-Baum (Dijkstra ab Depot). Kruskal ist deterministisch:
Kennzahlen laufen über 5 feste Instanzen (Seeds 100000-100004), Median mit 10./90. Perzentil.

- **Ersparnis** = 100 * (1 - Baumkosten / Kosten der Einzelanbindung): wie viel das gemeinsame Netz gegenüber getrennten Leitungen spart.
- **Umweg-Faktor** einer Filiale = Weglänge zum Depot ENTLANG des Baums / kürzester Weg im Kandidatengraphen (>= 1); der Kürzeste-Wege-Baum hat überall 1.
- **Betrachtete Kanten** = Anteil der Kanten, die Kruskal ansehen muss, bevor n - 1 Kanten angenommen sind.
- **Zeigerschritte** = Summe aller Schritte in den Union-Find-Suchen (Aufwandsmaß, unabhängig von Rechner und Sprache).
- **Instabilität** = Anteil der Läufe, in denen ein zufälliges Rauschen von +-1 % auf allen Kantenkosten den Baum ändert (Vorbote der Sensitivitäts-Stücks).
- **Anzahl MSTs** = Zahl aller kostenminimalen Spannbäume (Produkt über die Kostenklassen der Anzahl Spannbäume im jeweiligen Klassen-Multigraphen, Matrix-Tree-Satz)."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import groupby

import numpy as np

import kru_algorithm as A
import kru_constants as C
import kru_scenario as S
from kru_unionfind import MODES, UnionFind

INF = float("inf")


@dataclass(frozen=True)
class Settings:
    kind: str = "depot"
    n: int = C.DEFAULT_N
    k: int = C.DEFAULT_K
    terrain: float = C.DEFAULT_TERRAIN
    round_costs: bool = False
    seed: int = C.DEFAULT_SEED
    uf_mode: str = "full"
    sort: str = "all"


@lru_cache(maxsize=512)
def _generate(n, k, terrain, round_costs, seed):
    return S.generate(n, k, terrain, round_costs, seed)


@lru_cache(maxsize=64)
def _chain(n):
    return S.chain_instance(n)


def instance_of(settings):
    if settings.kind == "textbook":
        return S.textbook_instance()
    if settings.kind == "chain":
        return _chain(settings.n)
    return _generate(settings.n, settings.k, settings.terrain, settings.round_costs, settings.seed)


# --- Anzahl aller MSTs (Matrix-Tree-Satz je Kostenklasse) -------------------------------------------------------------------------------------------


def _bareiss_det(mat):
    """Exakte Determinante einer ganzzahligen Matrix (Bareiss, keine Rundung)."""
    m = [row[:] for row in mat]
    n = len(m)
    if n == 0:
        return 1
    sign, prev = 1, 1
    for i in range(n - 1):
        if m[i][i] == 0:
            swap = next((r for r in range(i + 1, n) if m[r][i] != 0), None)
            if swap is None:
                return 0
            m[i], m[swap] = m[swap], m[i]
            sign = -sign
        for r in range(i + 1, n):
            for c in range(i + 1, n):
                m[r][c] = (m[r][c] * m[i][i] - m[r][i] * m[i][c]) // prev
        prev = m[i][i]
    return sign * m[n - 1][n - 1]


def count_msts(n, edges):
    """Anzahl aller Spannbäume mit minimalen Kosten. Je Kostenklasse: Kanten zwischen bereits verschiedenen Komponenten; jede zusammenhängende Gruppe trägt die Zahl ihrer Spannbäume bei."""
    uf = UnionFind(n, "full")
    total = 1
    order = sorted(range(len(edges)), key=lambda i: edges[i][2])
    for _w, grp in groupby(order, key=lambda i: edges[i][2]):
        cls = []
        for i in grp:
            u, v, _ = edges[i]
            ru, rv = uf.find(u), uf.find(v)
            if ru != rv:
                cls.append((ru, rv))
        if not cls:
            continue
        nodes = sorted({x for e in cls for x in e})
        idx = {x: j for j, x in enumerate(nodes)}
        tmp = UnionFind(len(nodes), "full")
        for a, b in cls:
            tmp.union(idx[a], idx[b])
        groups = {}
        for x in range(len(nodes)):
            groups.setdefault(tmp.find(x), []).append(x)
        for members in groups.values():
            pos = {x: j for j, x in enumerate(members)}
            size = len(members)
            lap = [[0] * size for _ in range(size)]
            for a, b in cls:
                a, b = idx[a], idx[b]
                if a in pos:
                    i, j = pos[a], pos[b]
                    lap[i][i] += 1
                    lap[j][j] += 1
                    lap[i][j] -= 1
                    lap[j][i] -= 1
            total *= _bareiss_det([row[1:] for row in lap[1:]])
        for a, b in cls:
            uf.union(a, b)
    return total


# --- Analyse --------------------------------------------------------------------------------------------------------------------------------------


@dataclass
class Analysis:
    settings: Settings
    inst: object
    kruskal: A.KruskalResult
    prim_cost: float
    dist: list                # kürzeste Weglänge Depot -> Knoten
    spt_tree: list            # Kantenindizes des Kürzeste-Wege-Baums
    spt_cost: float
    tree_dist: list           # Weglänge Depot -> Knoten entlang des MST

    @property
    def mst_cost(self):
        return self.kruskal.cost

    @property
    def individual_cost(self):
        return float(sum(self.dist[1:]))

    @property
    def saving(self):
        return 100.0 * (1.0 - self.mst_cost / self.individual_cost) if self.individual_cost > 0 else 0.0

    @property
    def spt_over_mst(self):
        return 100.0 * (self.spt_cost / self.mst_cost - 1.0) if self.mst_cost > 0 else 0.0

    @property
    def stretch_by_node(self):
        return [1.0] + [self.tree_dist[x] / self.dist[x] if self.dist[x] > 0 else 1.0 for x in range(1, self.inst.n)]

    @property
    def stretch(self):
        return [self.tree_dist[x] / self.dist[x] for x in range(1, self.inst.n) if self.dist[x] > 0]

    @property
    def stretch_median(self):
        s = self.stretch
        return float(np.median(s)) if s else 1.0

    @property
    def stretch_max(self):
        s = self.stretch
        return float(max(s)) if s else 1.0

    @property
    def examined_share(self):
        return 100.0 * self.kruskal.examined / max(1, self.inst.m)

    @property
    def sorted_share(self):
        return 100.0 * self.kruskal.sorted_edges / max(1, self.inst.m)


def analyse(settings):
    inst = instance_of(settings)
    kr = A.kruskal(inst.n, inst.edges, settings.uf_mode, settings.sort)
    dist, _parent, pedge = A.shortest_path_tree(inst.n, inst.edges, inst.depot)
    spt = [pedge[x] for x in range(inst.n) if pedge[x] >= 0]
    return Analysis(settings, inst, kr, A.prim_cost(inst.n, inst.edges), dist, spt, float(sum(inst.edges[i][2] for i in spt)),
                    A.tree_distances(inst.n, inst.edges, kr.tree, inst.depot))


def uf_table(inst):
    """Union-Find-Stufen auf derselben Instanz: {Modus: (Zeigerschritte, längste Suche, größte Tiefe am Ende)}."""
    out = {}
    for mode in MODES:
        r = A.kruskal(inst.n, inst.edges, mode)
        out[mode] = (r.find_steps, r.max_find_len, max(r.depths) if r.depths else 0)
    return out


def instability(inst, eps=0.01, trials=20):
    """Anteil der Läufe, in denen +-eps Rauschen auf allen Kantenkosten die Kantenmenge des MST ändert."""
    base = {inst.edges[i][:2] for i in A.kruskal(inst.n, inst.edges).tree}
    changed = 0
    for t in range(trials):
        rng = np.random.default_rng([inst.seed, 31, t])
        noisy = [(u, v, w * (1.0 + eps * float(rng.uniform(-1.0, 1.0)))) for u, v, w in inst.edges]
        tree = {noisy[i][:2] for i in A.kruskal(inst.n, noisy).tree}
        changed += tree != base
    return 100.0 * changed / trials


# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------------------


def _stats(values):
    values = [v for v in values if not np.isnan(v) and v != INF]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90))


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    s0 = replace(base, **changes)
    rows = [analyse(replace(s0, seed=seed)) for seed in seeds]
    out = {"n_runs": len(rows)}
    for key, values in (
        ("mst_cost", [r.mst_cost for r in rows]),
        ("saving", [r.saving for r in rows]),
        ("spt_over_mst", [r.spt_over_mst for r in rows]),
        ("stretch_median", [r.stretch_median for r in rows]),
        ("stretch_max", [r.stretch_max for r in rows]),
        ("examined_share", [r.examined_share for r in rows]),
        ("edges", [float(r.inst.m) for r in rows]),
        ("find_steps", [float(r.kruskal.find_steps) for r in rows]),
        ("steps_per_edge", [r.kruskal.find_steps / max(1, r.kruskal.examined) for r in rows]),
        ("max_find_len", [float(r.kruskal.max_find_len) for r in rows]),
        ("sorted_share", [A.kruskal(r.inst.n, r.inst.edges, s0.uf_mode, "filter").sorted_edges * 100.0 / max(1, r.inst.m) for r in rows]),
        ("instability", [instability(r.inst) for r in rows]),
    ):
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = _stats(values)
    return out


SWEEP_VALUES = {"n": C.N_SWEEP, "k": C.K_SWEEP, "terrain": C.TERRAIN_SWEEP}
SWEEP_LABELS = {"n": "Filialen n", "k": "Nächste Nachbarn k (dichter = größer)", "terrain": "Geländezuschlag"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def uf_experiment(base=Settings(), sizes=C.UF_SIZES):
    """Zeigerschritte je Union-Find-Stufe über die Größe: auf Zufallsinstanzen (Median über die 5 festen Seeds, vollständiger Graph) und auf der Ketten-Instanz."""
    rows = []
    for n in sizes:
        rand = {mode: [] for mode in MODES}
        for seed in C.SWEEP_SEEDS:
            inst = _generate(n, 10 * n, base.terrain, False, seed)
            for mode, (steps, _l, _d) in uf_table(inst).items():
                rand[mode].append(steps)
        chain = uf_table(_chain(n))
        rows.append({"n": n, "random": {m: float(np.median(v)) for m, v in rand.items()}, "chain": {m: float(chain[m][0]) for m in MODES},
                     "chain_depth": {m: chain[m][2] for m in MODES}})
    return rows
