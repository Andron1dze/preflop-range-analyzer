import math

import pytest

from core.analysis.config import AnalysisConfig
from core.analysis.pipeline import run_analysis
from core.analysis.report import describe_coef, grid_layout, hand_deviations, load_report
from core.normalize.hand_class import ALL_HAND_CLASSES
from tests.analysis.factories import RFI, RFI_KEY, VS_OPEN, VS_OPEN_KEY, add_decisions, load_chart_set, sample_actions

OVERFOLD = {"fold": 0.7, "call": 0.2, "raise": 0.1}


# --- раскладка сетки 13×13 ---------------------------------------------------


def test_grid_layout_is_standard():
    grid = grid_layout()
    assert len(grid) == 13 and all(len(row) == 13 for row in grid)
    assert grid[0][:4] == ["AA", "AKs", "AQs", "AJs"]  # одномастные — справа сверху
    assert grid[1][:3] == ["AKo", "KK", "KQs"]  # разномастные — слева снизу
    assert grid[12][0] == "A2o" and grid[12][12] == "22"
    assert [grid[i][i] for i in range(13)] == ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22"]
    assert sorted(h for row in grid for h in row) == sorted(ALL_HAND_CLASSES)


# --- интерпретация β ---------------------------------------------------------


def test_describe_significant_overfold():
    text = describe_coef("continue_vs_fold", "position=BTN", -0.85, -1.1, -0.6)
    assert "position=BTN" in text
    assert "сбрасываете чаще солвера" in text
    assert "×0.43" in text  # шансы продолжить = exp(β)


def test_describe_significant_underfold_and_aggression():
    assert "продолжаете чаще солвера" in describe_coef("continue_vs_fold", "stack=<30", 0.5, 0.2, 0.8)
    assert "рейзите чаще солвера" in describe_coef("aggr_vs_call", "zone=pure", 0.4, 0.1, 0.7)
    assert "коллируете там, где солвер рейзит" in describe_coef("aggr_vs_call", "facing=3bet", -0.4, -0.7, -0.1)


def test_interval_containing_zero_is_not_significant():
    assert "нет значимого отклонения" in describe_coef("continue_vs_fold", "position=CO", -0.3, -0.8, 0.2)


def test_intercept_describes_reference_levels():
    assert "опорных уровнях" in describe_coef("continue_vs_fold", "intercept", -0.5, -0.8, -0.2)


def test_stage_coefficient_carries_icm_caveat():
    text = describe_coef("continue_vs_fold", "stage=bubble", -0.6, -0.9, -0.3)
    assert "частично ожидаемо из-за ICM" in text


def test_unknown_step_gets_generic_wording():
    assert "чаще солвера" in describe_coef("jam_vs_raise", "position=SB", 0.5, 0.1, 0.9)


# --- отчёт по прогону --------------------------------------------------------


@pytest.fixture
def analysed(session):
    chart_set, nodes = load_chart_set(session)
    add_decisions(
        session, nodes[VS_OPEN_KEY], sample_actions(OVERFOLD, 600, seed=1),
        position="BTN", line="UTG:F,UTG1:F,LJ:F,HJ:F,CO:R2.2", seed=1,
    )
    add_decisions(
        session, nodes[RFI_KEY], sample_actions(RFI, 400, seed=2),
        position="CO", line="UTG:F,UTG1:F,LJ:F,HJ:F", seed=2, stage=None,
    )
    run = run_analysis(
        session, chart_set_id=chart_set.id, mapping_version="m1", config=AnalysisConfig(), allow_synthetic=True
    )
    session.flush()
    return run, nodes


def test_report_rows_sorted_by_abs_z_with_leak_flags(session, analysed):
    run, nodes = analysed
    report = load_report(session, run.id)
    zs = [abs(row.top.z) for row in report.nodes]
    assert zs == sorted(zs, reverse=True)
    first = report.nodes[0]
    assert first.key == VS_OPEN_KEY and first.is_leak
    rfi_rows = [row for row in report.nodes if row.key == RFI_KEY]
    assert rfi_rows and not any(row.is_leak for row in rfi_rows)


def test_report_rows_expand_to_zones_and_groups(session, analysed):
    run, _ = analysed
    row = load_report(session, run.id).nodes[0]
    assert [z.zone for z in row.zones] == ["mix"]
    assert len(row.groups) > 3
    assert all(g.hand_group for g in row.groups)


def test_report_coefficients_grouped_by_step(session, analysed):
    run, _ = analysed
    coefs = load_report(session, run.id).coefs
    assert list(coefs) == ["continue_vs_fold", "aggr_vs_call"]
    assert coefs["continue_vs_fold"][0].feature == "intercept"
    assert all(c.text for c in coefs["continue_vs_fold"])


def test_missing_run_returns_none(session):
    assert load_report(session, 999) is None


def test_hand_deviations_for_node_action(session, analysed):
    run, nodes = analysed
    cells = hand_deviations(session, run.id, nodes[VS_OPEN_KEY].id, "fold")
    assert sum(c.n for c in cells.values()) == 600
    some = next(c for c in cells.values() if c.n)
    assert some.expected == pytest.approx(some.n * VS_OPEN["fold"])
    total_dev = sum(c.observed - c.expected for c in cells.values())
    assert total_dev > 0  # оверфолд
    assert all(math.isfinite(c.deviation) for c in cells.values() if c.n)
    assert all(c.deviation == 0 for c in cells.values() if not c.n)
