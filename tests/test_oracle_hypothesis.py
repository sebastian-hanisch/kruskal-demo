"""Property-basierte Ergänzung zum festen Orakeltest (test_oracle_kruskal.py): dasselbe Orakel (networkx-MST, SpanningTreeIterator, nachgebauter Union-Find-Lauf), aber mit
Hypothesis erzeugten Eingaben (Graphen mit 1-8 Knoten, bis 60 Kanten inklusive Parallelkanten und Schleifen, ganzzahlige und Gleitkomma-Kosten) und automatisch verkleinerten Gegenbeispielen.
Mehr als 32 Kanten sind nötig, damit Filter-Kruskal (FILTER_BASE = 32) überhaupt teilt. Deterministisch für die CI (derandomize, keine Beispieldatenbank)."""

import pytest

pytest.importorskip("hypothesis")
nx = pytest.importorskip("networkx")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import kru_algorithm as A  # noqa: E402
import kru_evaluation as ev  # noqa: E402
from kru_unionfind import MODES  # noqa: E402

CI = settings(max_examples=100, deadline=None, derandomize=True, database=None, suppress_health_check=[HealthCheck.too_slow])

MAX_N = 8
MAX_EDGES = 60


def _graph(n, edges):
    g = nx.MultiGraph()
    g.add_nodes_from(range(n))
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
    return g


@st.composite
def instances(draw, int_weights_only=False, simple=False, max_n=MAX_N, max_edges=MAX_EDGES):
    """(n, edges, tie): Kosten ganzzahlig (viele Gleichstände) oder Gleichkomma (>= 0)."""
    n = draw(st.integers(1, max_n))
    cost = st.integers(1, draw(st.sampled_from((1, 2, 3, 1000)))).map(float)
    if not int_weights_only and draw(st.booleans()):
        cost = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    node = st.integers(0, n - 1)
    if simple:
        pairs = st.lists(st.tuples(node, node).filter(lambda p: p[0] < p[1]), unique=True, max_size=max_edges)
        edges = [(u, v, draw(cost)) for u, v in draw(pairs)]
    else:
        min_size = draw(st.sampled_from((0, 0, 33)))                              # ein Teil der Fälle hat > 32 Kanten, damit Filter-Kruskal wirklich teilt
        edges = draw(st.lists(st.tuples(node, node, cost), min_size=min_size, max_size=max_edges))
    tie = draw(st.one_of(st.sampled_from(("lex", "reverse")), st.integers(0, 10 ** 6)))
    return n, edges, tie


@CI
@given(inst=instances(), sort=st.sampled_from(("all", "filter")), mode=st.sampled_from(MODES))
def test_kruskal_matches_networkx_on_generated_graphs(inst, sort, mode):
    n, edges, tie = inst
    g = _graph(n, edges)
    forest = nx.minimum_spanning_tree(g)
    r = A.kruskal(n, edges, mode, sort, tie)
    assert r.connected == nx.is_connected(g)
    assert len(r.tree) == forest.number_of_edges()
    assert r.cost == pytest.approx(forest.size(weight="weight"), abs=1e-9)
    assert len(set(r.tree)) == len(r.tree)
    assert nx.is_forest(_graph(n, [edges[i] for i in r.tree]))


@CI
@given(inst=instances(int_weights_only=True, simple=True, max_n=6, max_edges=15))
def test_mst_count_matches_the_networkx_enumeration(inst):
    n, edges, _tie = inst
    g = _graph(n, edges)
    if n < 2 or not nx.is_connected(g):
        return
    best = nx.minimum_spanning_tree(g).size(weight="weight")
    count = 0
    for tree in nx.SpanningTreeIterator(nx.Graph(g), minimum=True, weight="weight"):
        if tree.size(weight="weight") > best + 1e-9:
            break
        count += 1
    assert ev.count_msts(n, edges) == count


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


@CI
@given(inst=instances(), mode=st.sampled_from(MODES))
def test_union_find_pointer_steps_match_an_independent_replay(inst, mode):
    n, edges, _tie = inst
    r = A.kruskal(n, edges, mode)
    assert (r.find_steps, r.max_find_len) == _replay_find_steps(n, edges, mode)
