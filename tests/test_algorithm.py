"""Die zentrale Korrektheits-Kette für Kruskal: Ergebnis ist ein Spannbaum; Optimalität direkt über Schnitt- und Kreis-Eigenschaft, gegen Brute-Force-Aufzählung aller
Spannbäume, gegen scipy und ein unabhängiges Prim; Gleichstände; Union-Find-Stufen (gleicher Baum, Tiefenschranken, Zähler); Buchführung; Filter-Kruskal; Sonderfälle;
Referenzen (Dijkstra-Baum, Baumdistanzen)."""

import itertools
import math

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import kru_algorithm as A
import kru_scenario as S
import kru_unionfind as U

MODES = U.MODES


def _tree_adj(n, edges, tree):
    adj = {x: [] for x in range(n)}
    for i in tree:
        u, v, w = edges[i]
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def _is_spanning_tree(n, edges, tree):
    if len(tree) != n - 1 or len(set(tree)) != n - 1:
        return False
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for i in tree:
        u, v, _w = edges[i]
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return len({find(x) for x in range(n)}) == 1


def _path_max(adj, a, b):
    """Größte Kantenkosten auf dem (eindeutigen) Baumpfad a-b."""
    stack = [(a, -1, 0.0)]
    while stack:
        u, prev, mx = stack.pop()
        if u == b:
            return mx
        for v, w in adj[u]:
            if v != prev:
                stack.append((v, u, max(mx, w)))
    raise AssertionError("kein Pfad")


def _component(adj, start, banned):
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for v, _w in adj[u]:
            if (min(u, v), max(u, v)) != banned and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _brute_force_cost(n, edges):
    best = math.inf
    for combo in itertools.combinations(range(len(edges)), n - 1):
        if _is_spanning_tree(n, edges, list(combo)):
            best = min(best, sum(edges[i][2] for i in combo))
    return best


def _scipy_cost(n, edges):
    us, vs, ws = zip(*edges)
    mat = coo_matrix((ws, (us, vs)), shape=(n, n)).tocsr()
    return float(minimum_spanning_tree(mat).sum())


def _instances():
    for seed in range(6):
        for k in (3, 6, 100):
            for terrain in (0.0, 0.5):
                yield S.generate(20, k, terrain, False, seed)
    yield S.generate(20, 100, 0.3, True, 1)
    yield S.textbook_instance()
    yield S.chain_instance(20)


# --- Spannbaum und Optimalität ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("inst", list(_instances()))
def test_result_is_a_spanning_tree_with_recomputed_cost(inst):
    r = A.kruskal(inst.n, inst.edges)
    assert _is_spanning_tree(inst.n, inst.edges, r.tree) and r.connected
    assert r.cost == pytest.approx(sum(inst.edges[i][2] for i in r.tree))


@pytest.mark.parametrize("inst", list(_instances()))
def test_cut_and_cycle_properties_hold(inst):
    edges, n = inst.edges, inst.n
    tree = A.kruskal(n, edges).tree
    adj = _tree_adj(n, edges, tree)
    for i in tree:                                    # Schnitt-Eigenschaft: keine Kante über den Schnitt ist billiger als die Baumkante
        u, v, w = edges[i]
        side = _component(adj, u, (min(u, v), max(u, v)))
        for a, b, w2 in edges:
            if (a in side) != (b in side):
                assert w <= w2 + 1e-9
    in_tree = set(tree)
    for j, (u, v, w) in enumerate(edges):             # Kreis-Eigenschaft: eine Nichtbaumkante ist nie billiger als die teuerste Kante ihres Kreises
        if j not in in_tree:
            assert w >= _path_max(adj, u, v) - 1e-9


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("round_costs", [False, True])
def test_equals_brute_force_on_small_graphs(seed, round_costs):
    inst = S.generate(4, 100, 0.5, round_costs, seed)                    # 5 Knoten, vollständig: 10 Kanten
    assert A.kruskal(inst.n, inst.edges).cost == pytest.approx(_brute_force_cost(inst.n, inst.edges))


def test_textbook_example_by_hand():
    t = S.textbook_instance()
    r = A.kruskal(t.n, t.edges)
    assert r.cost == 15.0 and sorted(t.edges[i][:2] for i in r.tree) == [(0, 1), (1, 2), (1, 3), (2, 4)]
    assert [(t.edges[i][:2], acc) for i, acc, _c in r.steps] == [((1, 3), True), ((2, 4), True), ((0, 1), True), ((0, 3), False), ((1, 2), True)]
    assert r.examined == 5 and _brute_force_cost(t.n, t.edges) == 15.0


@pytest.mark.parametrize("inst", list(_instances()))
def test_matches_scipy_and_an_independent_prim(inst):
    r = A.kruskal(inst.n, inst.edges)
    assert r.cost == pytest.approx(_scipy_cost(inst.n, inst.edges)) and r.cost == pytest.approx(A.prim_cost(inst.n, inst.edges))


@pytest.mark.parametrize("seed", range(5))
def test_identical_tree_as_scipy_when_all_costs_are_distinct(seed):
    inst = S.generate(25, 8, 0.6, False, seed)
    assert len({w for _u, _v, w in inst.edges}) == inst.m
    us, vs, ws = zip(*inst.edges)
    tree_scipy = minimum_spanning_tree(coo_matrix((ws, (us, vs)), shape=(inst.n, inst.n)).tocsr()).tocoo()
    expected = sorted((min(a, b), max(a, b)) for a, b in zip(tree_scipy.row, tree_scipy.col))
    got = sorted(inst.edges[i][:2] for i in A.kruskal(inst.n, inst.edges).tree)
    assert got == expected


# --- Gleichstände -------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_tie_orders_change_the_tree_but_never_the_cost(seed):
    inst = S.generate(12, 100, 0.3, True, seed)
    costs, trees = set(), set()
    for tie in ("lex", "reverse", 0, 1, 2, 3, 4):
        r = A.kruskal(inst.n, inst.edges, tie=tie)
        assert _is_spanning_tree(inst.n, inst.edges, r.tree)
        costs.add(round(r.cost, 9))
        trees.add(tuple(sorted(r.tree)))
    assert len(costs) == 1 and len(trees) >= 1


def test_complete_graph_with_equal_costs_reaches_many_different_trees():
    n = 5
    edges = tuple((u, v, 1.0) for u in range(n) for v in range(u + 1, n))
    trees = {tuple(sorted(A.kruskal(n, edges, tie=s).tree)) for s in range(200)}
    assert 10 < len(trees) <= n ** (n - 2)                              # Cayley: 125 Bäume, jede Tie-Reihenfolge trifft einen davon


def test_the_default_tie_break_is_deterministic():
    inst = S.generate(20, 100, 0.3, True, 2)
    assert A.kruskal(inst.n, inst.edges).tree == A.kruskal(inst.n, inst.edges).tree


# --- Union-Find ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("inst", list(_instances())[:16])
def test_all_union_find_modes_give_the_same_tree_and_steps(inst):
    base = A.kruskal(inst.n, inst.edges, "full")
    for mode in MODES:
        r = A.kruskal(inst.n, inst.edges, mode)
        assert (r.tree, r.steps, r.examined, r.cost) == (base.tree, base.steps, base.examined, base.cost)


@pytest.mark.parametrize("seed", range(5))
def test_union_find_matches_a_naive_component_list(seed):
    rng = np.random.default_rng(seed)
    for mode in MODES:
        uf = U.UnionFind(30, mode)
        comp = list(range(30))
        for _ in range(100):
            a, b = int(rng.integers(30)), int(rng.integers(30))
            merged = uf.union(a, b)
            assert merged == (comp[a] != comp[b])
            if merged:
                old = comp[b]
                comp = [comp[a] if c == old else c for c in comp]
            assert uf.find(a) == uf.find(b) if comp[a] == comp[b] else uf.find(a) != uf.find(b)
        assert uf.components == len(set(comp))


def test_rank_bound_and_naive_chain_depth():
    n = 64
    chain = S.chain_instance(n)
    depth = {mode: max(A.kruskal(chain.n, chain.edges, mode).depths) for mode in MODES}
    assert depth["naive"] == n - 1                                       # Kette
    assert depth["rank"] <= math.ceil(math.log2(n)) and depth["full"] <= math.ceil(math.log2(n))
    for seed in range(5):
        inst = S.generate(60, 100, 0.3, False, seed)
        assert max(A.kruskal(inst.n, inst.edges, "rank").depths) <= math.ceil(math.log2(inst.n))


def test_union_find_counters_are_consistent():
    inst = S.generate(40, 100, 0.3, False, 3)
    for mode in MODES:
        r = A.kruskal(inst.n, inst.edges, mode)
        assert r.finds == 2 * r.examined and r.unions == inst.n - 1 and r.find_steps >= 0 and r.max_find_len <= r.find_steps
    chain = S.chain_instance(50)
    naive, full = A.kruskal(chain.n, chain.edges, "naive"), A.kruskal(chain.n, chain.edges, "full")
    assert naive.find_steps > 5 * full.find_steps and naive.max_find_len == chain.n - 2


def test_invalid_modes_are_rejected():
    with pytest.raises(ValueError):
        U.UnionFind(3, "x")
    with pytest.raises(ValueError):
        A.kruskal(3, [(0, 1, 1.0)], sort="x")


# --- Buchführung --------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("inst", list(_instances())[:16])
def test_bookkeeping_of_steps_and_early_stop(inst):
    r = A.kruskal(inst.n, inst.edges)
    assert len(r.steps) == r.examined <= inst.m and len({i for i, _a, _c in r.steps}) == r.examined
    accepted = 0
    for i, acc, comps in r.steps:
        accepted += acc
        assert comps == inst.n - accepted
    assert accepted == inst.n - 1 and r.steps[-1][1] is True                      # Stopp sofort nach der (n-1)-ten Kante
    ws = [inst.edges[i][2] for i, _a, _c in r.steps]
    assert ws == sorted(ws)


def test_rejections_close_a_cycle_and_acceptances_join_two_components():
    inst = S.generate(20, 100, 0.3, False, 1)
    r = A.kruskal(inst.n, inst.edges)
    uf = U.UnionFind(inst.n, "full")
    for i, acc, _c in r.steps:
        u, v, _w = inst.edges[i]
        assert (uf.find(u) != uf.find(v)) == acc
        uf.union(u, v)


# --- Filter-Kruskal -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("k", [6, 100])
def test_filter_kruskal_gives_the_same_tree_for_distinct_costs(seed, k):
    inst = S.generate(60, k, 0.6, False, seed)
    a, b = A.kruskal(inst.n, inst.edges), A.kruskal(inst.n, inst.edges, sort="filter")
    assert sorted(b.tree) == sorted(a.tree) and b.cost == pytest.approx(a.cost) and b.sorted_edges <= inst.m


@pytest.mark.parametrize("seed", range(6))
def test_filter_kruskal_with_ties_keeps_the_cost(seed):
    inst = S.generate(60, 100, 0.3, True, seed)
    b = A.kruskal(inst.n, inst.edges, sort="filter")
    assert _is_spanning_tree(inst.n, inst.edges, b.tree) and b.cost == pytest.approx(A.kruskal(inst.n, inst.edges).cost)


def test_filter_kruskal_sorts_fewer_edges_on_a_dense_graph():
    inst = S.generate(120, 1000, 0.3, False, 2)
    plain, filt = A.kruskal(inst.n, inst.edges), A.kruskal(inst.n, inst.edges, sort="filter")
    assert plain.sorted_edges == inst.m and filt.sorted_edges < 0.6 * inst.m and filt.filtered_out > 0


# --- Sonderfälle --------------------------------------------------------------------------------------------------------------------------------


def test_single_node_two_nodes_and_tree_input():
    one = A.kruskal(1, [])
    assert one.tree == [] and one.cost == 0.0 and one.connected and one.examined == 0
    two = A.kruskal(2, [(0, 1, 3.0), (0, 1, 5.0)])
    assert two.cost == 3.0 and two.examined == 1
    path = [(i, i + 1, float(i + 1)) for i in range(9)]
    r = A.kruskal(10, path)
    assert len(r.tree) == 9 and r.examined == 9 and r.cost == 45.0


def test_disconnected_graph_gives_a_forest_and_is_flagged():
    edges = [(0, 1, 1.0), (1, 2, 2.0), (3, 4, 1.0)]
    r = A.kruskal(5, edges)
    assert not r.connected and len(r.tree) == 3 and r.examined == 3 and r.cost == 4.0


def test_complete_graph_with_equal_costs_and_cayley_count():
    n = 5
    edges = tuple((u, v, 1.0) for u in range(n) for v in range(u + 1, n))
    trees = [c for c in itertools.combinations(range(len(edges)), n - 1) if _is_spanning_tree(n, edges, list(c))]
    assert len(trees) == 125 and A.kruskal(n, edges).cost == 4.0


# --- Referenzen ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_dijkstra_tree_matches_floyd_warshall_and_mst_is_no_costlier_than_it(seed):
    inst = S.generate(15, 5, 0.4, False, seed)
    n = inst.n
    d = np.full((n, n), np.inf)
    np.fill_diagonal(d, 0.0)
    for u, v, w in inst.edges:
        d[u, v] = d[v, u] = w
    for k in range(n):
        d = np.minimum(d, d[:, [k]] + d[[k], :])
    dist, parent, pedge = A.shortest_path_tree(n, inst.edges, 0)
    assert np.allclose(dist, d[0])
    spt = [pedge[x] for x in range(1, n)]
    assert _is_spanning_tree(n, inst.edges, spt) and sum(inst.edges[i][2] for i in spt) >= A.kruskal(n, inst.edges).cost - 1e-9
    td = A.tree_distances(n, inst.edges, spt, 0)
    assert np.allclose(td, dist)


def test_tree_distances_on_a_path_and_unreachable_nodes():
    edges = [(0, 1, 2.0), (1, 2, 3.0), (3, 4, 1.0)]
    td = A.tree_distances(5, edges, [0, 1], 0)
    assert td[:3] == [0.0, 2.0, 5.0] and td[3] == math.inf and td[4] == math.inf
