"""Die Instanz dieser Demo: ein Depot (Werk) und n Filialen (Kunden) als Punkte auf einer Karte; gesucht wird ein Leitungsnetz (Fernwärme, Glasfaser,
Sammelleitung), das alle verbindet. Kantenkosten = Trassenlänge = euklidischer Abstand mal Geländefaktor u in [1, 1 + Zuschlag] (damit der Baum nicht rein
geometrisch ist). Der Faktor hängt nur vom Knotenpaar und vom Seed ab, NICHT vom Kandidatengraphen - wer k ändert, streicht Kanten, ändert aber keine Kosten.

Kandidatengraph: `k >= n_gesamt - 1` = vollständig (dicht); sonst die k nächsten Nachbarn je Knoten (Vereinigung beider Richtungen), bei Bedarf um die kürzeste
Zwischenkante ergänzt, bis der Graph zusammenhängt (ohne diese Garantie wäre "kein Spannbaum" der Normalfall bei kleinem k, nicht die Ausnahme).

`round_costs` rundet die Kosten auf ganze km (mindestens 1): das erzeugt bewusst viele Gleichstände. Zwei handgebaute Fixtures: ein Lehrbuchbeispiel (5 Knoten) und eine
Ketten-Instanz, an der Union-Find ohne Gegenmaßnahmen quadratisch wird."""

from dataclasses import dataclass

import numpy as np

import kru_constants as C

INSTANCE_KINDS = ("depot", "textbook", "chain")


@dataclass(frozen=True)
class Instance:
    xy: np.ndarray                 # (N, 2), Knoten 0 = Depot
    edges: tuple                   # ((u, v, w), ...) mit u < v, sortiert nach (u, v)
    kind: str = "depot"
    k: int = 0
    terrain: float = 0.0
    round_costs: bool = False
    seed: int = 0
    labels: object = None          # Knotennamen der Fixtures

    @property
    def n(self):
        return len(self.xy)

    @property
    def m(self):
        return len(self.edges)

    @property
    def depot(self):
        return 0


def _components_connect(n, edge_set, dist):
    """Ergänzt `edge_set` um die kürzeste Zwischenkante zweier Komponenten, bis alles zusammenhängt."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edge_set:
        parent[find(u)] = find(v)
    while True:
        roots = [find(x) for x in range(n)]
        if len(set(roots)) == 1:
            return
        best = None
        for u in range(n):
            for v in range(u + 1, n):
                if roots[u] != roots[v] and (best is None or (dist[u, v], u, v) < best[0]):
                    best = ((dist[u, v], u, v), u, v)
        _, u, v = best
        edge_set.add((u, v))
        parent[find(u)] = find(v)


def generate(n_customers=C.DEFAULT_N, k=C.DEFAULT_K, terrain=C.DEFAULT_TERRAIN, round_costs=False, seed=C.DEFAULT_SEED):
    """Depot + `n_customers` Filialen; Knoten 0 ist das Depot."""
    n = int(n_customers) + 1
    rng = np.random.default_rng([int(seed), 909])
    xy = np.vstack([np.array([C.DEPOT_XY]), rng.uniform(0.0, C.AREA, size=(n - 1, 2))])
    factor = rng.uniform(1.0, 1.0 + float(terrain), size=(n, n))
    factor = np.triu(factor, 1) + np.triu(factor, 1).T
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    cost = dist * np.where(factor > 0, factor, 1.0)
    if round_costs:
        cost = np.maximum(C.ROUND_UNIT, np.round(cost / C.ROUND_UNIT) * C.ROUND_UNIT)
    if k >= n - 1:
        edge_set = {(u, v) for u in range(n) for v in range(u + 1, n)}
    else:
        edge_set = set()
        order = np.argsort(dist + np.diag(np.full(n, np.inf)), axis=1, kind="stable")
        for u in range(n):
            for v in order[u, :k]:
                edge_set.add((min(u, int(v)), max(u, int(v))))
        _components_connect(n, edge_set, dist)
    edges = tuple((u, v, float(cost[u, v])) for u, v in sorted(edge_set))
    return Instance(xy, edges, "depot", int(k), float(terrain), bool(round_costs), int(seed))


# --- Handgebaute Fixtures -----------------------------------------------------------------------------------------------------------------------

TEXTBOOK_XY = [(10.0, 15.0), (40.0, 15.0), (70.0, 15.0), (25.0, 45.0), (55.0, 45.0)]
TEXTBOOK_EDGES = [(0, 1, 4.0), (1, 2, 6.0), (0, 3, 5.0), (1, 3, 2.0), (1, 4, 7.0), (2, 4, 3.0), (3, 4, 8.0)]


def textbook_instance():
    """Fünf Knoten A-E, sieben Kanten; der MST hat Kosten 15 (B-D 2, C-E 3, A-B 4, B-C 6), A-D (5) schließt einen Kreis und wird verworfen."""
    return Instance(np.array(TEXTBOOK_XY), tuple(sorted(TEXTBOOK_EDGES)), "textbook", labels=("A", "B", "C", "D", "E"))


def chain_instance(n_nodes=30):
    """Ketten-Instanz: Kanten (i, i+1) mit Kosten 2i+1 und Kanten (0, i+1) mit Kosten 2i+2 (i >= 1). Kruskal nimmt die Kettenkante an und verwirft danach die Kante von
    Knoten 0 - bei naivem Union-Find liegt Knoten 0 dann am Ende einer immer längeren Kette, jede Suche kostet so viele Schritte wie die Kette lang ist."""
    n = int(n_nodes)
    angles = np.linspace(0.0, np.pi, n)
    xy = np.stack([C.AREA / 2 + 40 * np.cos(angles), C.AREA / 2 + 40 * np.sin(angles)], axis=1)
    edges = [(i, i + 1, float(2 * i + 1)) for i in range(n - 1)]
    edges += [(0, i + 1, float(2 * i + 2)) for i in range(1, n - 1)]
    return Instance(xy, tuple(sorted(edges)), "chain")
