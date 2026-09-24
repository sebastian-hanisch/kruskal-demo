"""Jede Zahl aus README, App-Texten und kru_constants.py, nachgerechnet über die echten Auswertungsfunktionen (ev.sweep/ev.run_config/ev.uf_experiment) auf den 5 festen Instanzen
(Seeds 100000-100004), 30 Filialen, k = 6, Geländezuschlag 0.3, exakte Kosten, sofern nicht anders angegeben. Kruskal ist deterministisch; Toleranzen sind großzügig gegenüber Rundung."""

from functools import lru_cache

import pytest

import kru_constants as C
import kru_evaluation as ev

BASE = ev.Settings()


@lru_cache(maxsize=None)
def _rows(param):
    return ev.sweep(param, BASE)


@lru_cache(maxsize=None)
def _uf():
    return ev.uf_experiment(BASE)


def _col(param, key):
    return [r[key] for r in _rows(param)]


def _at(param, value):
    return next(r for r in _rows(param) if r["value"] == value)


def _close(values, expected, tol):
    assert len(values) == len(expected)
    for v, e in zip(values, expected):
        assert v == pytest.approx(e, abs=tol), (values, expected)


def test_default_configuration_numbers():
    r = ev.run_config(BASE)
    assert r["saving"] == pytest.approx(73.7, abs=0.1) and r["spt_over_mst"] == pytest.approx(47.2, abs=0.1)
    assert r["stretch_median"] == pytest.approx(1.37, abs=0.01) and r["stretch_max"] == pytest.approx(3.32, abs=0.01)
    assert r["examined_share"] == pytest.approx(62.9, abs=0.1) and r["sorted_share"] == pytest.approx(41.0, abs=0.1) and r["instability"] == 30.0


def test_saving_grows_with_the_number_of_customers():
    _close(_col("n", "saving"), [52.1, 67.5, 77.6, 86.2, 90.4], 0.1)
    assert _col("n", "value") == [10, 20, 40, 80, 160] and _col("n", "saving") == sorted(_col("n", "saving"))


def test_terrain_barely_changes_the_saving():
    assert _col("terrain", "value") == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    _close(_col("terrain", "saving"), [74.3, 73.8, 73.7, 73.8, 73.8, 73.8], 0.1)
    assert max(_col("terrain", "saving")) - min(_col("terrain", "saving")) < 0.7


def test_detours_grow_with_the_size_and_the_shortest_path_tree_costs_more():
    assert _at("n", 160)["stretch_max"] == pytest.approx(7.95, abs=0.02)
    _close(_col("k", "spt_over_mst"), [24.7, 35.3, 47.2, 75.7, 127.4, 137.7], 0.1)
    assert _col("k", "value") == [3, 4, 6, 10, 20, 1000]


def test_denser_graphs_mean_fewer_examined_and_sorted_edges():
    _close(_col("k", "examined_share"), [88.7, 79.0, 62.9, 42.4, 22.2, 17.9], 0.1)
    _close(_col("k", "sorted_share"), [78.1, 71.6, 41.0, 27.3, 13.5, 13.3], 0.1)
    assert _col("k", "examined_share") == sorted(_col("k", "examined_share"), reverse=True)


def test_instability_over_the_size():
    _close(_col("n", "instability"), [0.0, 0.0, 30.0, 60.0, 70.0], 0.1)


def test_rounded_costs_make_the_tree_far_less_stable():
    r = ev.run_config(BASE, round_costs=True)
    assert r["instability"] == 80.0 and r["saving"] == pytest.approx(73.85, abs=0.1)


def test_union_find_steps_on_random_instances():
    rows = _uf()
    assert [r["n"] for r in rows] == list(C.UF_SIZES)
    for mode, expected in (("naive", [151, 778, 1237, 4001]), ("compress", [69, 282, 438, 945]), ("rank", [105, 407, 753, 1889]), ("full", [66, 250, 385, 881])):
        _close([r["random"][mode] for r in rows], expected, 0.6)
    assert 4 < rows[-1]["random"]["naive"] / rows[-1]["random"]["full"] < 5


def test_union_find_steps_on_the_chain_instance():
    rows = _uf()
    for mode, expected in (("naive", [170, 740, 3080, 12560]), ("compress", [17, 37, 77, 157]), ("rank", [51, 111, 231, 471]), ("full", [51, 111, 231, 471])):
        _close([r["chain"][mode] for r in rows], expected, 0.6)
    assert [r["chain_depth"]["naive"] for r in rows] == [n - 1 for n in C.UF_SIZES] and all(r["chain_depth"]["full"] == 1 for r in rows)


def test_compression_alone_is_close_to_the_full_variant_and_rank_alone_is_not():
    r = _uf()[-1]["random"]
    assert r["compress"] < 1.1 * r["full"] and r["rank"] > 1.9 * r["full"]
