"""Unabhängiges Orakel (networkx, eigener Wiederholungslauf): Kruskal in allen Varianten gegen networkx-MST auf Zufallsgraphen mit vielen Gleichständen und Wäldern; Zahl der kostenminimalen
Spannbäume gegen die Aufzählung von networkx (SpanningTreeIterator); Zeigerschritte des Union-Find gegen eine nachgebaute Referenz."""

import random

import pytest

nx = pytest.importorskip("networkx")

import kru_algorithm as A  # noqa: E402
import kru_evaluation as ev  # noqa: E402
from kru_unionfind import MODES  # noqa: E402


def _graph(n, edges):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
    return g


def _random_edges(rng, n, p, wmax):
    return [(u, v, float(rng.randint(1, wmax))) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def test_all_variants_match_networkx_on_random_graphs_with_ties_and_forests():
    rng = random.Random(1)
    for t in range(150):
        n = rng.randint(1, 12)
        edges = _random_edges(rng, n, rng.choice([0.2, 0.5, 1.0]), rng.choice([1, 2, 3, 1000]))
        g = _graph(n, edges)
        forest = nx.minimum_spanning_tree(g)
        for sort in ("all", "filter"):
            for tie in ("lex", "reverse", t):
                for mode in MODES:
                    r = A.kruskal(n, edges, mode, sort, tie)
                    assert r.connected == nx.is_connected(g)
                    assert len(r.tree) == forest.number_of_edges()
                    assert r.cost == pytest.approx(forest.size(weight="weight"))
                    assert nx.is_forest(_graph(n, [edges[i] for i in r.tree]))


def test_mst_count_matches_the_networkx_enumeration():
    rng = random.Random(2)
    done = 0
    while done < 40:
        n = rng.randint(3, 8)
        edges = _random_edges(rng, n, 0.5, 3)
        g = _graph(n, edges)
        if not nx.is_connected(g):
            continue
        best = nx.minimum_spanning_tree(g).size(weight="weight")
        count = 0
        for tree in nx.SpanningTreeIterator(g, minimum=True, weight="weight"):
            if tree.size(weight="weight") > best + 1e-9:
                break
            count += 1
        assert ev.count_msts(n, edges) == count
        done += 1


def _replay_find_steps(n, edges, mode):
    parent, rank = list(range(n)), [0] * n
    steps, longest = 0, 0

    def find(x):
        nonlocal steps, longest
        s = 0
        while parent[x] != x:
            if mode in ("compress", "full"):
                parent[x] = parent[parent[x]]
            x = parent[x]
            s += 1
        steps += s
        longest = max(longest, s)
        return x

    accepted = 0
    for i in sorted(range(len(edges)), key=lambda i: (edges[i][2], i)):
        if accepted >= n - 1:
            break
        a, b = find(edges[i][0]), find(edges[i][1])
        if a == b:
            continue
        if mode in ("rank", "full") and rank[a] > rank[b]:
            a, b = b, a
        parent[a] = b
        if mode in ("rank", "full") and rank[a] == rank[b]:
            rank[b] += 1
        accepted += 1
    return steps, longest


@pytest.mark.parametrize("mode", MODES)
def test_union_find_pointer_steps_match_an_independent_replay(mode):
    import kru_scenario as S

    for inst in (S.textbook_instance(), S.chain_instance(40), S.generate(25, 1000, 0.3, False, 100000), S.generate(30, 6, 0.3, True, 7)):
        r = A.kruskal(inst.n, inst.edges, mode)
        assert (r.find_steps, r.max_find_len) == _replay_find_steps(inst.n, inst.edges, mode)
