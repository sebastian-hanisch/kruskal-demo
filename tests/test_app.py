"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt und jede Kante, alle Instanz-Typen, Randwerte, Würfel-Knopf, Permalink-Grenzen, Instanzwechsel, Union-Find-Experiment,
Gleichstände und Sweeps auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kru_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="kru_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label.startswith(label))


def test_default_run_has_no_exception_and_shows_the_four_metrics():
    at = _run()
    _ok(at)
    assert {"Baumkosten (MST)", "Ersparnis", "Umweg-Faktor", "Betrachtete Kanten"} <= {m.label for m in at.metric}
    assert _metric(at, "Baumkosten") == "466.63" and _metric(at, "Ersparnis") == "72.4 %" and _metric(at, "Umweg") == "1.26 / 3.54" and _metric(at, "Betrachtete") == "90 / 113"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["kind_select"] == p["kind"] and at.session_state["k_select"] == p["k"] and at.session_state["uf_select"] == p["uf_mode"] and at.session_state["sort_select"] == p["sort"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
def test_every_step_runs_for_every_kind(step):
    for kind in C.KINDS:
        at = _run(kind_select=kind, step=step)
        _ok(at)
        assert at.get("plotly_chart") and at.session_state["kru_step"] == step


def test_edge_slider_walks_through_all_examined_edges_and_survives_an_instance_change():
    at = _run(step=2)
    _ok(at)
    slider = at.slider(key="kru_edge")
    assert slider.max == 90
    slider.set_value(slider.max).run()
    _ok(at)
    assert at.session_state["kru_edge"] == 90
    at.session_state["kind_select"] = "textbook"
    at.run()
    _ok(at)
    assert at.session_state["kru_edge"] <= 5


@pytest.mark.parametrize("kw", [
    dict(n_slider=C.N_MIN), dict(n_slider=C.N_MAX), dict(n_slider=C.N_MAX, k_select=1000), dict(n_slider=C.N_MIN, k_select=3), dict(k_select=C.K_OPTIONS[0]), dict(k_select=C.K_OPTIONS[-1]),
    dict(terrain_select=C.TERRAIN_OPTIONS[0]), dict(terrain_select=C.TERRAIN_OPTIONS[-1]), dict(round_select=True), dict(uf_select="naive"), dict(uf_select="compress"), dict(uf_select="rank"),
    dict(sort_select="filter"), dict(kind_select="chain", n_slider=C.N_MIN), dict(kind_select="chain", n_slider=C.N_MAX, uf_select="naive"),
])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))
    _ok(_run(step=2, **kw))
    _ok(_run(step=3, **kw))


def test_union_find_and_sort_switches_never_change_the_tree_cost():
    base = _metric(_run(), "Baumkosten")
    for kw in (dict(uf_select="naive"), dict(uf_select="rank"), dict(sort_select="filter")):
        assert _metric(_run(**kw), "Baumkosten") == base


def test_abstract_chain_instance_hides_map_metrics():
    at = _run(kind_select="chain", n_slider=30)
    assert _metric(at, "Ersparnis") == "-" and _metric(at, "Umweg") == "-"


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(n="9999", k="7", terrain="0.35", round="maybe", uf="x", sort="x", kind="nope").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert ss["n_slider"] == C.N_MAX
    assert (ss["k_select"], ss["terrain_select"], ss["round_select"], ss["uf_select"], ss["sort_select"], ss["kind_select"]) == (C.DEFAULT_K, C.DEFAULT_TERRAIN, False, "full", "all", "depot")


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(kind="depot", n="50", k="1000", terrain="0.6", round="true", uf="rank", sort="filter", seed="7").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["n_slider"], ss["k_select"], ss["terrain_select"], ss["round_select"], ss["uf_select"], ss["sort_select"], ss["seed_input"]) == (50, 1000, 0.6, True, "rank", "filter", 7)


def test_sidebar_hides_map_only_controls_for_the_fixtures():
    for kind in ("textbook", "chain"):
        at = _run(kind_select=kind)
        _ok(at)
        assert not any(s.key == "k_select" for s in at.select_slider) and not any(n.key == "seed_input" for n in at.number_input)
    assert any(s.key == "n_slider" for s in _run(kind_select="chain").slider) and not any(s.key == "n_slider" for s in _run(kind_select="textbook").slider)


def test_changing_the_instance_while_on_step_two_does_not_crash():
    at = _run(step=2)
    _ok(at)
    at.session_state["n_slider"] = C.N_MIN
    at.run()
    _ok(at)
    at.session_state["kind_select"] = "chain"
    at.run()
    _ok(at)


@pytest.mark.parametrize("param", ["n", "k", "terrain"])
@pytest.mark.parametrize("metric", ["saving", "stretch", "examined", "uf", "instability"])
def test_sweeps_run_on_demand_for_every_metric(param, metric):
    at = _run(n_slider=15, sweep_metric=metric)
    at.selectbox(key="sweep_select").set_value(param).run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_union_find_experiment_runs_on_demand():
    at = _run(n_slider=15)
    next(b for b in at.button if b.key == "uf_start").click().run()
    _ok(at)
    assert len(at.get("plotly_chart")) >= 4


def test_ties_and_stability_run_on_demand():
    at = _run(round_select=True)
    next(b for b in at.button if b.key == "tie_start").click().run()
    _ok(at)
    assert _metric(at, "Kostengleiche Bäume") == "4" and _metric(at, "Baum ändert sich") == "65 %"


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Wer setzt an" in m.value and "Der billigste Baum ist ein gutes Netz" in m.value for m in at.markdown)
