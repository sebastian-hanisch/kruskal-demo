"""Presets: Vollständigkeit, gültige Werte, der Median der Ersparnis bleibt bei den Karten-Presets in der gemessenen Spannweite über die 5 festen Instanzen, und jedes
Preset zeigt, was sein Name und sein Hilfetext sagen."""

import pytest

import kru_algorithm as A
import kru_constants as C
import kru_evaluation as ev
import kru_presets as P


def _settings(p):
    return ev.Settings(kind=p["kind"], n=p["n"], k=p["k"], terrain=p["terrain"], round_costs=p["round_costs"], seed=p["seed"], uf_mode=p["uf_mode"], sort=p["sort"])


def _analyse(name):
    return ev.analyse(_settings(C.PRESETS[name]))


def test_every_preset_has_help_and_the_map_presets_a_band():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 9
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]
    assert set(C.PRESET_EXPECTED_BANDS) <= {n for n, p in C.PRESETS.items() if p["kind"] == "depot"}


def test_preset_values_are_valid_members_of_the_controls():
    for p in C.PRESETS.values():
        assert p["kind"] in C.KINDS and p["k"] in C.K_OPTIONS and p["terrain"] in C.TERRAIN_OPTIONS and p["uf_mode"] in C.UF_MODES and p["sort"] in C.SORTS
        assert C.N_MIN <= p["n"] <= C.N_MAX and 0 <= p["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


@pytest.mark.parametrize("name", list(C.PRESET_EXPECTED_BANDS))
def test_map_preset_median_saving_stays_in_its_measured_band(name):
    lo, hi = C.PRESET_EXPECTED_BANDS[name]
    assert lo <= ev.run_config(_settings(C.PRESETS[name]))["saving"] <= hi


def test_standard_preset_numbers():
    a = _analyse("Standardfall (Voreinstellung)")
    assert a.mst_cost == pytest.approx(466.63, abs=0.01) and a.saving == pytest.approx(72.4, abs=0.05) and a.spt_over_mst == pytest.approx(57.3, abs=0.1)
    assert (a.kruskal.examined, a.inst.m) == (90, 113) and (round(a.stretch_median, 2), round(a.stretch_max, 2)) == (1.26, 3.54)
    assert ev.instability(a.inst) == 35.0 and a.mst_cost == pytest.approx(a.prim_cost)


def test_thin_network_preset():
    a, s = _analyse("Dünnes Netz (k = 3)"), _analyse("Standardfall (Voreinstellung)")
    assert (a.kruskal.examined, a.inst.m) == (58, 59) and a.mst_cost == pytest.approx(482.90, abs=0.01) and a.spt_over_mst == pytest.approx(24.5, abs=0.1)
    assert a.mst_cost / s.mst_cost == pytest.approx(1.035, abs=0.002)


def test_complete_graph_preset_gives_the_same_tree_and_sees_few_edges():
    a, s = _analyse("Vollständiger Graph"), _analyse("Standardfall (Voreinstellung)")
    assert a.inst.m == 465 and a.kruskal.examined == 109 and a.mst_cost == pytest.approx(s.mst_cost) and a.spt_over_mst == pytest.approx(122.7, abs=0.1)
    assert A.kruskal(a.inst.n, a.inst.edges, sort="filter").sorted_edges == 56


def test_many_customers_preset():
    a = _analyse("Viele Filialen (n = 160)")
    assert a.saving == pytest.approx(89.6, abs=0.05) and (round(a.stretch_median, 2), round(a.stretch_max, 2)) == (2.38, 9.61)


def test_ties_presets_count_the_equal_cost_trees():
    a = _analyse("Gleichstände (gerundet)")
    assert a.mst_cost == 467.0 and ev.count_msts(a.inst.n, a.inst.edges) == 4 and ev.instability(a.inst) == 65.0
    trees = {tuple(sorted(A.kruskal(a.inst.n, a.inst.edges, tie=t).tree)) for t in range(50)}
    assert 1 < len(trees) <= 4
    b = _analyse("Sehr viele gleich billige Bäume")
    assert ev.count_msts(b.inst.n, b.inst.edges) == 1536


def test_filter_preset_sorts_a_tiny_share_of_a_dense_graph():
    a = _analyse("Filter-Kruskal (dicht)")
    plain = A.kruskal(a.inst.n, a.inst.edges)
    assert a.inst.m == 7260 and a.kruskal.examined == 153 and a.kruskal.sorted_edges == 161 and a.kruskal.filtered_out == 1085 and plain.cost == pytest.approx(a.kruskal.cost)


def test_chain_preset_naive_union_find_is_quadratic():
    a = _analyse("Ketten-Instanz (naives Union-Find)")
    t = ev.uf_table(a.inst)
    assert (t["naive"][0], t["compress"][0], t["rank"][0], t["full"][0]) == (1710, 57, 171, 171) and a.kruskal.max_find_len == 58 and t["naive"][2] == 59


def test_textbook_preset():
    a = _analyse("Lehrbuchbeispiel")
    assert a.mst_cost == 15.0 and a.kruskal.examined == 5 and a.inst.m == 7


def test_bounds_and_permalink_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_permalink_casters_accept_members_and_reject_everything_else():
    c = P.SETTING_SPECS
    assert c["kind_select"].caster("chain") == "chain" and c["k_select"].caster("1000") == 1000 and c["terrain_select"].caster("0.4") == 0.4
    assert c["round_select"].caster("true") is True and c["round_select"].caster("False") is False and c["uf_select"].caster("rank") == "rank" and c["sort_select"].caster("filter") == "filter"
    for key, bad in (("kind_select", "x"), ("k_select", "7"), ("terrain_select", "0.35"), ("terrain_select", "x"), ("round_select", "maybe"), ("uf_select", "x"), ("sort_select", "x")):
        with pytest.raises(ValueError):
            c[key].caster(bad)
