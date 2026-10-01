import numpy as np
import pytest
from sqlalchemy import select

from core.db.models import AnalysisRun, ModelCoef
from core.models.action_tree import DEFAULT_TREE, StepInput, TreeStep, load_tree, step_dataset
from core.models.offset_logit import FitResult, logit, save_coefs

CONTINUE, AGGRESSION = DEFAULT_TREE
EPS = 1e-4
THREE_WAY = frozenset({"fold", "call", "raise"})
RFI = frozenset({"fold", "raise"})


def row(strategy, action, node_actions=THREE_WAY):
    return StepInput(strategy=strategy, action=action, node_actions=node_actions)


# --- дерево из конфига -------------------------------------------------------


def test_default_tree_from_spec():
    assert [s.name for s in DEFAULT_TREE] == ["continue_vs_fold", "aggr_vs_call"]
    assert CONTINUE.population is None
    assert CONTINUE.positive == frozenset({"check", "call", "raise", "allin"})
    assert AGGRESSION.population == frozenset({"check", "call", "raise", "allin"})
    assert AGGRESSION.positive == frozenset({"raise", "allin"})


def test_tree_is_loaded_from_config():
    tree = load_tree({
        "steps": [
            {"name": "continue_vs_fold", "population": None, "positive": ["check", "call", "raise", "allin"]},
            {"name": "aggr_vs_call", "population": ["check", "call", "raise", "allin"], "positive": ["raise", "allin"]},
            {"name": "jam_vs_raise", "population": ["raise", "allin"], "positive": ["allin"]},
        ]
    })
    assert tree[:2] == DEFAULT_TREE
    assert tree[2] == TreeStep("jam_vs_raise", frozenset({"raise", "allin"}), frozenset({"allin"}))


@pytest.mark.parametrize(
    "config",
    [
        {"steps": []},
        {"steps": [{"name": "x", "population": None, "positive": ["limp"]}]},
        {"steps": [{"name": "x", "population": ["call"], "positive": ["raise"]}]},  # positive ⊄ population
        {"steps": [{"population": None, "positive": ["call"]}]},
    ],
)
def test_invalid_tree_config_is_rejected(config):
    with pytest.raises(ValueError):
        load_tree(config)


# --- шаг 1: продолжить или сбросить ------------------------------------------


def test_continue_step_offset_is_one_minus_fold():
    data = step_dataset(CONTINUE, [row({"fold": 0.4, "call": 0.35, "raise": 0.25}, "fold")], eps=EPS)
    assert data.rows == [0]
    assert data.y.tolist() == [0.0]
    assert data.offset[0] == pytest.approx(logit(np.array([0.6]))[0])


def test_pure_frequencies_are_clipped():
    data = step_dataset(CONTINUE, [row({"fold": 1.0}, "call"), row({"raise": 1.0}, "raise")], eps=EPS)
    assert data.offset == pytest.approx(logit(np.array([EPS, 1 - EPS])))


# --- шаг 2: агрессия или колл при условии продолжения ------------------------


def test_aggression_step_is_conditional_on_continuing():
    decisions = [
        row({"fold": 0.4, "call": 0.35, "raise": 0.25}, "raise"),
        row({"fold": 0.4, "call": 0.35, "raise": 0.25}, "fold"),  # не продолжил — в шаг 2 не входит
        row({"fold": 0.4, "call": 0.35, "raise": 0.25}, "call"),
    ]
    data = step_dataset(AGGRESSION, decisions, eps=EPS)
    assert data.rows == [0, 2]
    assert data.y.tolist() == [1.0, 0.0]
    assert data.offset == pytest.approx(logit(np.array([0.25 / 0.6] * 2)))


def test_undefined_conditional_probability_is_excluded():
    # GTO всегда сбрасывает: условная частота агрессии не определена.
    data = step_dataset(AGGRESSION, [row({"fold": 1.0}, "call")], eps=EPS)
    assert data.rows == []


def test_rfi_node_has_degenerate_aggression_step():
    # В узле открытия нет колла: «агрессия или колл» вырождается.
    data = step_dataset(AGGRESSION, [row({"fold": 0.5, "raise": 0.5}, "raise", RFI)], eps=EPS)
    assert data.rows == []
    # А шаг 1 в RFI работает: «открыться или сбросить».
    assert step_dataset(CONTINUE, [row({"fold": 0.5, "raise": 0.5}, "raise", RFI)], eps=EPS).rows == [0]


def test_pure_raise_in_node_with_call_is_not_degenerate():
    # AA всегда 3-бетит, но колл в узле есть — строка информативна.
    data = step_dataset(AGGRESSION, [row({"raise": 1.0}, "call")], eps=EPS)
    assert data.rows == [0]
    assert data.offset[0] == pytest.approx(logit(np.array([1 - EPS]))[0])


def test_allin_counts_as_aggression():
    data = step_dataset(AGGRESSION, [row({"call": 0.5, "allin": 0.5}, "allin", frozenset({"fold", "call", "allin"}))], eps=EPS)
    assert data.y.tolist() == [1.0]


# --- запись коэффициентов ----------------------------------------------------


def test_coefficients_and_intervals_are_saved(session):
    run = AnalysisRun(mapping_version="m1", chart_set_version="c1", params={}, status="running")
    session.add(run)
    session.flush()
    fit = FitResult(
        beta=np.array([0.1, -0.5]),
        se=np.array([0.05, 0.2]),
        ci_low=np.array([0.0, -0.9]),
        ci_high=np.array([0.2, -0.1]),
        converged=True,
        iterations=5,
    )
    save_coefs(session, run.id, "continue_vs_fold", ["intercept", "stack<30bb"], fit)
    session.flush()

    rows = session.scalars(select(ModelCoef).order_by(ModelCoef.id)).all()
    assert [(r.tree_step, r.feature, r.beta, r.se, r.ci_low, r.ci_high) for r in rows] == [
        ("continue_vs_fold", "intercept", 0.1, 0.05, 0.0, 0.2),
        ("continue_vs_fold", "stack<30bb", -0.5, 0.2, -0.9, -0.1),
    ]


def test_feature_names_must_match_coefficients(session):
    fit = FitResult(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), True, 1)
    with pytest.raises(ValueError):
        save_coefs(session, 1, "continue_vs_fold", ["intercept"], fit)
