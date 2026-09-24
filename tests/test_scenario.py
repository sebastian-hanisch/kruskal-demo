import numpy as np
import pytest

import kru_constants as C
import kru_scenario as S


def _connected(inst):
    parent = list(range(inst.n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for u, v, _w in inst.edges:
        parent[find(u)] = find(v)
    return len({find(x) for x in range(inst.n)}) == 1


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("k", [3, 4, 10])
def test_generated_graph_is_always_connected_and_edges_are_canonical(seed, k):
    inst = S.generate(25, k, 0.3, False, seed)
    assert inst.n == 26 and _connected(inst)
    assert all(u < v for u, v, _w in inst.edges) and list(inst.edges) == sorted(inst.edges, key=lambda e: (e[0], e[1]))
    assert len({(u, v) for u, v, _w in inst.edges}) == inst.m and all(w > 0 for _u, _v, w in inst.edges)


def test_dense_graph_is_complete_and_sparse_graph_is_smaller():
    dense = S.generate(20, 100, 0.3, False, 1)
    sparse = S.generate(20, 4, 0.3, False, 1)
    assert dense.m == 21 * 20 // 2 and sparse.m < dense.m and sparse.m >= 20


def test_changing_k_only_removes_edges_and_never_changes_a_cost():
    dense = {(u, v): w for u, v, w in S.generate(25, 100, 0.5, False, 7).edges}
    for k in (3, 5, 8):
        for u, v, w in S.generate(25, k, 0.5, False, 7).edges:
            assert dense[(u, v)] == pytest.approx(w)


def test_costs_are_euclidean_distance_times_a_terrain_factor_in_range():
    inst = S.generate(20, 100, 0.4, False, 3)
    for u, v, w in inst.edges:
        d = float(np.hypot(*(inst.xy[u] - inst.xy[v])))
        assert d - 1e-9 <= w <= d * 1.4 + 1e-9


def test_zero_terrain_makes_costs_exactly_euclidean():
    inst = S.generate(15, 100, 0.0, False, 3)
    assert all(w == pytest.approx(float(np.hypot(*(inst.xy[u] - inst.xy[v])))) for u, v, w in inst.edges)


def test_rounding_creates_ties_and_integer_costs():
    inst = S.generate(30, 100, 0.3, True, 5)
    ws = [w for _u, _v, w in inst.edges]
    assert all(w == round(w) and w >= 1 for w in ws) and len(set(ws)) < len(ws)
    assert len({w for _u, _v, w in S.generate(30, 100, 0.3, False, 5).edges}) == S.generate(30, 100, 0.3, False, 5).m


def test_generation_is_deterministic_and_seed_dependent():
    a, b, c = S.generate(20, 5, 0.3, False, 4), S.generate(20, 5, 0.3, False, 4), S.generate(20, 5, 0.3, False, 5)
    assert a.edges == b.edges and np.array_equal(a.xy, b.xy) and a.edges != c.edges


def test_depot_is_node_zero_at_the_fixed_position_and_customers_lie_in_the_area():
    inst = S.generate(40, 5, 0.3, False, 9)
    assert tuple(inst.xy[0]) == C.DEPOT_XY and inst.depot == 0
    assert inst.xy.min() >= 0 and inst.xy.max() <= C.AREA


def test_textbook_fixture_shape():
    t = S.textbook_instance()
    assert t.n == 5 and t.m == 7 and t.labels == ("A", "B", "C", "D", "E") and _connected(t)


def test_chain_fixture_shape_and_cost_order():
    c = S.chain_instance(12)
    assert c.n == 12 and c.m == (11) + (10) and _connected(c)
    w = {(u, v): x for u, v, x in c.edges}
    assert w[(0, 1)] == 1 and w[(1, 2)] == 3 and w[(0, 2)] == 4 and w[(2, 3)] == 5 and w[(0, 3)] == 6
