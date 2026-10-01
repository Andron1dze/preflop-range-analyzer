import pytest
from sqlalchemy import select

from core.analysis.config import AnalysisConfig
from core.analysis.pipeline import run_analysis
from core.db.models import AnalysisRun, ChartSet, ModelCoef, NodeStat
from tests.analysis.factories import (
    RFI,
    RFI_KEY,
    VS_OPEN,
    VS_OPEN_KEY,
    add_decisions,
    load_chart_set,
    sample_actions,
)

# Герой перефолживает против открытия CO: fold 0.7 вместо 0.5.
OVERFOLD = {"fold": 0.7, "call": 0.2, "raise": 0.1}


@pytest.fixture
def setup(session):
    chart_set, nodes = load_chart_set(session)
    add_decisions(
        session, nodes[VS_OPEN_KEY], sample_actions(OVERFOLD, 600, seed=1),
        position="BTN", line="UTG:F,UTG1:F,LJ:F,HJ:F,CO:R2.2", seed=1,
    )
    add_decisions(
        session, nodes[RFI_KEY], sample_actions(RFI, 400, seed=2),
        position="CO", line="UTG:F,UTG1:F,LJ:F,HJ:F", seed=2, stage=None,
    )
    return chart_set, nodes


def run(session, chart_set, **kwargs):
    analysis = run_analysis(
        session, chart_set_id=chart_set.id, mapping_version="m1", config=AnalysisConfig(),
        allow_synthetic=True, **kwargs,
    )
    session.flush()
    return analysis


def top_rows(session, run_id):
    rows = session.scalars(
        select(NodeStat).where(NodeStat.run_id == run_id, NodeStat.zone.is_(None), NodeStat.hand_group.is_(None))
    ).all()
    return {(r.node_id, r.action): r for r in rows}


# --- запись прогона ----------------------------------------------------------


def test_run_is_recorded_with_versions_and_params(session, setup):
    chart_set, _ = setup
    analysis = run(session, chart_set)
    row = session.get(AnalysisRun, analysis.id)
    assert row.status == "done"
    assert row.mapping_version == "m1"
    assert row.chart_set_version == "synthetic-test@1"
    assert row.params["stats"]["z_threshold"] == 2.0
    assert row.params["allow_synthetic"] is True


def test_synthetic_chart_set_is_rejected_by_default(session, setup):
    chart_set, _ = setup
    with pytest.raises(ValueError, match="not eligible"):
        run_analysis(session, chart_set_id=chart_set.id, mapping_version="m1", config=AnalysisConfig())
    assert session.scalars(select(AnalysisRun)).all() == []


def test_unknown_chart_set_is_rejected(session, setup):
    with pytest.raises(ValueError, match="chart set"):
        run_analysis(session, chart_set_id=999, mapping_version="m1", config=AnalysisConfig(), allow_synthetic=True)


# --- статистика по узлам -----------------------------------------------------


def test_every_node_action_gets_a_top_level_row(session, setup):
    chart_set, nodes = setup
    rows = top_rows(session, run(session, chart_set).id)
    vs_open, rfi = nodes[VS_OPEN_KEY].id, nodes[RFI_KEY].id
    assert set(rows) == {
        (vs_open, "fold"), (vs_open, "call"), (vs_open, "raise"),
        (rfi, "fold"), (rfi, "raise"),
    }
    assert rows[(vs_open, "fold")].n == 600
    assert rows[(vs_open, "fold")].expected == pytest.approx(600 * VS_OPEN["fold"])


def test_overfold_is_detected_and_gto_play_is_not(session, setup):
    chart_set, nodes = setup
    rows = top_rows(session, run(session, chart_set).id)
    assert rows[(nodes[VS_OPEN_KEY].id, "fold")].z > 2
    assert abs(rows[(nodes[RFI_KEY].id, "fold")].z) < 3


def test_bh_q_only_on_top_level_rows(session, setup):
    chart_set, _ = setup
    run_id = run(session, chart_set).id
    rows = session.scalars(select(NodeStat).where(NodeStat.run_id == run_id)).all()
    top = [r for r in rows if r.zone is None and r.hand_group is None]
    assert all(r.bh_q is not None for r in top)
    assert all(r.bh_q is None for r in rows if r not in top)
    assert {r.zone for r in rows if r.zone} == {"mix"}
    assert len({r.hand_group for r in rows if r.hand_group}) > 3


def test_unknown_stage_share(session, setup):
    chart_set, nodes = setup
    rows = top_rows(session, run(session, chart_set).id)
    assert rows[(nodes[VS_OPEN_KEY].id, "fold")].unknown_stage_share == 0.0
    assert rows[(nodes[RFI_KEY].id, "fold")].unknown_stage_share == 1.0


def test_only_mapped_weighted_hero_decisions_are_used(session, setup):
    chart_set, nodes = setup
    node = nodes[VS_OPEN_KEY]
    line = "UTG:F,UTG1:F,LJ:F,HJ:F,CO:R2.2"
    add_decisions(session, node, ["fold"] * 50, position="BTN", line=line, actor="villain", seed=3)
    add_decisions(session, node, ["fold"] * 50, position="BTN", line=line, mapping_version="m0", seed=4)
    add_decisions(session, node, ["fold"] * 50, position="BTN", line=line, weight=0.0, seed=5)
    rows = top_rows(session, run(session, chart_set).id)
    assert rows[(node.id, "fold")].n == 600


# --- регрессия ---------------------------------------------------------------


def coefs(session, run_id, step):
    rows = session.scalars(select(ModelCoef).where(ModelCoef.run_id == run_id, ModelCoef.tree_step == step))
    return {r.feature: r for r in rows}


def test_regression_coefficients_are_saved_per_tree_step(session, setup):
    chart_set, _ = setup
    run_id = run(session, chart_set).id
    continue_step = coefs(session, run_id, "continue_vs_fold")
    assert continue_step["intercept"].ci_low < continue_step["intercept"].ci_high
    # RFI-решения (CO) входят в шаг 1, CO — опорный уровень, BTN — отдельный признак.
    assert "position=BTN" in continue_step
    assert coefs(session, run_id, "aggr_vs_call")  # шаг 2 — только узел против открытия


def test_overfold_shows_as_less_continuing_on_btn(session, setup):
    # Только позиция и без штрафа: CO играет по GTO (β ≈ 0), BTN продолжает реже:
    # logit(0.3) − logit(0.5) ≈ −0.85.
    chart_set, _ = setup
    analysis = run_analysis(
        session, chart_set_id=chart_set.id, mapping_version="m1",
        config=AnalysisConfig(features=("position",), l2=0.0), allow_synthetic=True,
    )
    session.flush()
    step = coefs(session, analysis.id, "continue_vs_fold")
    assert set(step) == {"intercept", "position=BTN"}
    assert step["intercept"].ci_low < 0 < step["intercept"].ci_high
    assert step["position=BTN"].beta == pytest.approx(-0.847, abs=0.3)
    assert step["position=BTN"].ci_high < 0


def test_rerun_with_same_inputs_is_identical(session, setup):
    chart_set, _ = setup
    first, second = run(session, chart_set).id, run(session, chart_set).id

    def snapshot(run_id):
        stats = session.execute(
            select(NodeStat.node_id, NodeStat.action, NodeStat.zone, NodeStat.hand_group, NodeStat.n,
                   NodeStat.expected, NodeStat.observed, NodeStat.z, NodeStat.bh_q)
            .where(NodeStat.run_id == run_id).order_by(NodeStat.id)
        ).all()
        model = session.execute(
            select(ModelCoef.tree_step, ModelCoef.feature, ModelCoef.beta, ModelCoef.se)
            .where(ModelCoef.run_id == run_id).order_by(ModelCoef.id)
        ).all()
        return stats, model

    assert snapshot(first) == snapshot(second)


def test_failed_run_is_marked_failed(session, setup, monkeypatch):
    chart_set, _ = setup

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("core.analysis.pipeline.fit_irls", boom)
    with pytest.raises(RuntimeError):
        run(session, chart_set)
    assert session.scalars(select(AnalysisRun.status)).all() == ["failed"]
    # Статистика успела посчитаться до падения регрессии, но частичных строк не остаётся.
    assert session.scalars(select(NodeStat)).all() == []


def test_chart_set_must_belong_to_nodes_of_mapping(session, setup):
    # Решения, привязанные к узлам другого набора чартов, в прогон не попадают.
    chart_set, _ = setup
    other = ChartSet(model="chipev", ante="bb:1", source="other", version="1", eligible_for_analysis=True)
    session.add(other)
    session.flush()
    analysis = run_analysis(session, chart_set_id=other.id, mapping_version="m1", config=AnalysisConfig())
    session.flush()
    assert session.scalars(select(NodeStat).where(NodeStat.run_id == analysis.id)).all() == []
