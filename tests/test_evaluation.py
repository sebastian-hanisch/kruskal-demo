"""Auswertung: MST-Zählung (Matrix-Tree je Kostenklasse) gegen Aufzählung, Analysis-Felder gegen unabhängige Neuberechnung, uf_table, instability, run_config/sweep/uf_experiment."""

import itertools
import math

import numpy as np
import pytest

import kru_algorithm as A
import kru_constants as C
import kru_evaluation as ev
import kru_scenario as S
from kru_unionfind import MODES

SMALL = dict(seeds=C.SWEEP_SEEDS[:2])


def _spanning_tree_count(n, edges, only_min=False):
    best, count = math.inf, 0
    for combo in itertools.combinations(range(len(edges)), n - 1):
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        ok = True
        for i in combo:
            u, v, _w = edges[i]
            ru, rv = find(u), find(v)
            if ru == rv:
                ok = False
                break
            parent[ru] = rv
        if not ok:
            continue
        cost = sum(edges[i][2] for i in combo)
        if not only_min:
            count += 1
        elif cost < best - 1e-9:
            best, count = cost, 1
        elif abs(cost - best) <= 1e-9:
            count += 1
    return count


@pytest.mark.parametrize("seed", range(8))
def test_count_msts_matches_brute_force_on_small_rounded_graphs(seed):
    inst = S.generate(5, 100, 0.3, True, seed)                          # 6 Knoten, vollständig, gerundet: viele Gleichstände
    assert ev.count_msts(inst.n, inst.edges) == _spanning_tree_count(inst.n, inst.edges, only_min=True)


def test_count_msts_special_cases():
    n = 5
    equal = tuple((u, v, 1.0) for u in range(n) for v in range(u + 1, n))
    assert ev.count_msts(n, equal) == 125 == _spanning_tree_count(n, equal)                       # Cayley n^(n-2)
    assert ev.count_msts(S.textbook_instance().n, S.textbook_instance().edges) == 1
    path = [(i, i + 1, float(i + 1)) for i in range(6)]
    assert ev.count_msts(7, path) == 1
    cycle = [(i, (i + 1) % 6, 1.0) for i in range(6)]
    cycle = [(min(u, v), max(u, v), w) for u, v, w in cycle]
    assert ev.count_msts(6, cycle) == 6                                                             # Kreis mit gleichen Kosten: eine Kante weglassen
    assert ev.count_msts(1, []) == 1


@pytest.mark.parametrize("seed", range(6))
def test_bareiss_determinant_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    for size in (1, 2, 4, 6):
        m = rng.integers(-5, 6, size=(size, size))
        assert ev._bareiss_det(m.tolist()) == round(float(np.linalg.det(m)))
    assert ev._bareiss_det([]) == 1 and ev._bareiss_det([[0, 1], [1, 0]]) == -1 and ev._bareiss_det([[0, 0], [0, 0]]) == 0


def test_default_settings_come_from_the_constants_and_are_members_of_the_controls():
    s = ev.Settings()
    assert (s.n, s.k, s.terrain, s.seed) == (C.DEFAULT_N, C.DEFAULT_K, C.DEFAULT_TERRAIN, C.DEFAULT_SEED)
    assert s.k in C.K_OPTIONS and s.terrain in C.TERRAIN_OPTIONS and s.uf_mode in C.UF_MODES and s.sort in C.SORTS and s.kind in C.KINDS


@pytest.mark.parametrize("seed", [35, 100001])
def test_analysis_fields_match_an_independent_recomputation(seed):
    a = ev.analyse(ev.Settings(seed=seed))
    inst = S.generate(30, 6, 0.3, False, seed)
    kr = A.kruskal(inst.n, inst.edges)
    assert a.mst_cost == pytest.approx(kr.cost) and a.prim_cost == pytest.approx(kr.cost)
    d = A.shortest_path_tree(inst.n, inst.edges, 0)[0]
    assert a.individual_cost == pytest.approx(sum(d[1:])) and a.saving == pytest.approx(100 * (1 - kr.cost / sum(d[1:])))
    td = A.tree_distances(inst.n, inst.edges, kr.tree)
    assert min(a.stretch) >= 1.0 - 1e-9 and a.stretch_max == pytest.approx(max(td[x] / d[x] for x in range(1, inst.n)))
    assert a.spt_cost >= a.mst_cost - 1e-9 and a.spt_over_mst == pytest.approx(100 * (a.spt_cost / a.mst_cost - 1))
    assert a.examined_share == pytest.approx(100 * kr.examined / inst.m) and len(a.stretch_by_node) == inst.n and a.stretch_by_node[0] == 1.0


def test_shortest_path_tree_stretch_is_one_and_the_mst_is_never_costlier():
    for seed in range(5):
        a = ev.analyse(ev.Settings(seed=seed))
        spt_tree_dist = A.tree_distances(a.inst.n, a.inst.edges, a.spt_tree)
        assert np.allclose(spt_tree_dist, a.dist) and a.mst_cost <= a.spt_cost + 1e-9 <= a.individual_cost + 1e-9


def test_kinds_produce_the_expected_instances():
    assert ev.analyse(ev.Settings(kind="textbook")).mst_cost == 15.0
    c = ev.analyse(ev.Settings(kind="chain", n=40)).inst
    assert c.n == 40 and c.kind == "chain"
    assert ev.analyse(ev.Settings(uf_mode="naive")).mst_cost == ev.analyse(ev.Settings(uf_mode="full")).mst_cost


def test_uf_table_and_naive_versus_full_on_the_chain():
    t = ev.uf_table(ev.instance_of(ev.Settings(kind="chain", n=60)))
    assert set(t) == set(MODES) and t["naive"][2] == 59 and t["naive"][0] > 5 * t["full"][0] and t["rank"][2] == 1


def test_instability_is_deterministic_and_between_zero_and_hundred():
    inst = ev.instance_of(ev.Settings())
    a, b = ev.instability(inst), ev.instability(inst)
    assert a == b and 0.0 <= a <= 100.0
    assert ev.instability(S.textbook_instance()) == 0.0                                                # weit auseinanderliegende Kosten


def test_run_config_keys_ranges_and_base_seed_independence():
    r = ev.run_config(ev.Settings(seed=1), **SMALL)
    assert r["n_runs"] == 2
    for key in ("mst_cost", "saving", "spt_over_mst", "stretch_median", "stretch_max", "examined_share", "edges", "find_steps", "steps_per_edge", "max_find_len", "sorted_share", "instability"):
        assert r[f"{key}_lo"] <= r[key] <= r[f"{key}_hi"]
    assert 0 < r["saving"] < 100 and r["stretch_median"] >= 1.0 and 0 < r["examined_share"] <= 100 and r["sorted_share"] <= 100
    assert r == pytest.approx(ev.run_config(ev.Settings(seed=999), **SMALL))


def test_sweep_rows_and_labels():
    rows = ev.sweep("n", ev.Settings(), values=(10, 20))
    assert [r["value"] for r in rows] == [10, 20] and rows[1]["saving"] > rows[0]["saving"]
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS) == {"n", "k", "terrain"}


def test_uf_experiment_shape_and_ordering():
    rows = ev.uf_experiment(ev.Settings(), sizes=(20, 40))
    assert [r["n"] for r in rows] == [20, 40]
    for r in rows:
        assert set(r["random"]) == set(r["chain"]) == set(MODES)
        assert r["chain"]["naive"] > r["chain"]["compress"] and r["chain_depth"]["naive"] == r["n"] - 1
