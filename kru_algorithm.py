"""Kruskal-Algorithmus und Referenzen.

Kruskal (1956): Kanten nach Kosten sortieren; jede Kante wird angenommen, wenn sie zwei verschiedene Komponenten verbindet (Schnitt-Eigenschaft: die billigste Kante über einen
Schnitt gehört zu einem MST), und verworfen, wenn sie einen Kreis schließen würde (Kreis-Eigenschaft: die teuerste Kante eines Kreises gehört zu keinem MST). Ob zwei Knoten
schon verbunden sind, beantwortet Union-Find (`kru_unionfind.py`). Stopp, sobald n - 1 Kanten angenommen sind.

Sortierung: `sort="all"` sortiert alle Kanten; `sort="filter"` ist Filter-Kruskal (Osipov, Sanders & Singler 2009): die Kanten werden quicksort-artig am Median dreier Kanten geteilt, die
leichte Hälfte zuerst bearbeitet, und aus der schweren Hälfte fliegen vor dem Sortieren alle Kanten, die schon innerhalb einer Komponente liegen - sie werden nie sortiert.

Gleichstände: `tie` legt fest, wie Kanten gleicher Kosten geordnet werden ("lex" = nach (u, v), "reverse" = umgekehrt, oder ein ganzzahliger Seed = feste Zufallsreihenfolge). Die Kosten
eines MST sind davon unabhängig, der Baum selbst nicht immer."""

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from kru_unionfind import UnionFind

FILTER_BASE = 32                 # ab dieser Größe wird nicht mehr geteilt, sondern sortiert


@dataclass
class KruskalResult:
    tree: list                                     # Indizes der angenommenen Kanten (in Annahmereihenfolge)
    cost: float
    steps: list = field(default_factory=list)      # je betrachtete Kante: (Kantenindex, angenommen, Komponentenzahl danach)
    examined: int = 0                              # betrachtete Kanten bis zum Stopp
    sorted_edges: int = 0                          # Zahl der Kanten, die tatsächlich sortiert wurden
    filtered_out: int = 0                          # Filter-Kruskal: vor dem Sortieren verworfene Kanten
    finds: int = 0
    find_steps: int = 0
    max_find_len: int = 0
    unions: int = 0
    depths: list = field(default_factory=list)     # Tiefe jedes Knotens im Union-Find-Wald am Ende
    connected: bool = True


def _tie_key(m, tie):
    """Rangfolge innerhalb gleicher Kosten: Liste `rank[i]` für Kante i."""
    if tie == "lex":
        return list(range(m))
    if tie == "reverse":
        return list(range(m - 1, -1, -1))
    rng = np.random.default_rng([int(tie), 4711])
    return [int(x) for x in rng.permutation(m)]


def kruskal(n, edges, uf_mode="full", sort="all", tie="lex"):
    """`edges` = Sequenz (u, v, w). Gibt `KruskalResult` zurück; bei unzusammenhängendem Graphen ein Wald (`connected=False`)."""
    if sort not in ("all", "filter"):
        raise ValueError(f"unbekannte Sortierung {sort}")
    m = len(edges)
    rank = _tie_key(m, tie)
    key = lambda i: (edges[i][2], rank[i])
    uf = UnionFind(n, uf_mode)
    res = KruskalResult([], 0.0)
    need = n - 1

    def process(order):
        for i in order:
            if len(res.tree) >= need:
                return True
            res.examined += 1
            u, v, w = edges[i]
            if uf.union(u, v):
                res.tree.append(i)
                res.cost += w
                res.steps.append((i, True, uf.components))
            else:
                res.steps.append((i, False, uf.components))
        return len(res.tree) >= need

    if sort == "all":
        order = sorted(range(m), key=key)
        res.sorted_edges = m
        process(order)
    else:
        def filter_kruskal(idx):
            if len(res.tree) >= need or not idx:
                return
            if len(idx) <= FILTER_BASE:
                order = sorted(idx, key=key)
                res.sorted_edges += len(order)
                process(order)
                return
            trio = sorted((idx[0], idx[len(idx) // 2], idx[-1]), key=key)
            pivot = key(trio[1])
            light = [i for i in idx if key(i) <= pivot]
            heavy = [i for i in idx if key(i) > pivot]
            if not heavy or not light:                      # Pivot teilt nicht: sortieren, sonst Endlosschleife
                order = sorted(idx, key=key)
                res.sorted_edges += len(order)
                process(order)
                return
            filter_kruskal(light)
            if len(res.tree) >= need:
                return
            kept = []
            for i in heavy:
                u, v, _w = edges[i]
                if uf.find(u) != uf.find(v):
                    kept.append(i)
            res.filtered_out += len(heavy) - len(kept)
            filter_kruskal(kept)

        filter_kruskal(list(range(m)))
    res.finds, res.find_steps, res.max_find_len, res.unions = uf.finds, uf.find_steps, uf.max_find_len, uf.unions
    res.depths = uf.depths()
    res.connected = uf.components == 1
    return res


# --- Referenzen (Kontrollrechnungen und Vergleichsgrößen) ---------------------------------------------------------------------------------------


def prim_cost(n, edges):
    """Unabhängige Kontrollrechnung: Prim mit binärem Heap ab Knoten 0 (Kosten des MST; Graph muss zusammenhängen)."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((w, v))
        adj[v].append((w, u))
    seen = [False] * n
    heap = [(0.0, 0)]
    total, count = 0.0, 0
    while heap and count < n:
        w, u = heapq.heappop(heap)
        if seen[u]:
            continue
        seen[u] = True
        total += w
        count += 1
        for w2, v in adj[u]:
            if not seen[v]:
                heapq.heappush(heap, (w2, v))
    return total if count == n else math.inf


def shortest_path_tree(n, edges, root=0):
    """Dijkstra ab `root`: (dist, parent, parent_edge); nicht erreichbare Knoten haben dist = inf."""
    adj = [[] for _ in range(n)]
    for idx, (u, v, w) in enumerate(edges):
        adj[u].append((v, w, idx))
        adj[v].append((u, w, idx))
    dist = [math.inf] * n
    parent = [-1] * n
    parent_edge = [-1] * n
    dist[root] = 0.0
    heap = [(0.0, root)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w, idx in adj[u]:
            nd = d + w
            if nd < dist[v] or (nd == dist[v] and parent[v] > u >= 0):        # Gleichstand: kleinerer Elternknoten (deterministisch)
                dist[v], parent[v], parent_edge[v] = nd, u, idx
                heapq.heappush(heap, (nd, v))
    return dist, parent, parent_edge


def tree_distances(n, edges, tree, root=0):
    """Weglänge jedes Knotens zur Wurzel ENTLANG des Baums `tree` (Liste von Kantenindizes); nicht erreichbare: inf."""
    adj = [[] for _ in range(n)]
    for idx in tree:
        u, v, w = edges[idx]
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist = [math.inf] * n
    dist[root] = 0.0
    stack = [root]
    while stack:
        u = stack.pop()
        for v, w in adj[u]:
            if dist[v] == math.inf:
                dist[v] = dist[u] + w
                stack.append(v)
    return dist
