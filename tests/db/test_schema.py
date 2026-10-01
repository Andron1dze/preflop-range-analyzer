import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from core.db.models import ChartSet, Decision, Hand

EXPECTED_COLUMNS = {
    "hands": {"id", "source", "raw_text", "tournament_id", "level", "ante", "imported_at"},
    "decisions": {
        "id", "hand_id", "actor", "position", "hole_cards", "hand_class", "action",
        "size_bb", "size_pot", "eff_stack_bb", "line", "stage", "stage_source", "ev_loss",
    },
    "chart_sets": {"id", "model", "ante", "source", "version"},
    "nodes": {"id", "chart_set_id", "key", "stack_bucket", "sizing_bucket", "sizings"},
    "node_strategies": {"node_id", "hand_class", "action", "freq"},
    "decision_node_map": {"decision_id", "node_id", "mapping_version", "distance", "weight"},
    "analysis_runs": {"id", "started_at", "mapping_version", "chart_set_version", "params", "status"},
    "node_stats": {
        "run_id", "node_id", "zone", "hand_group", "n", "expected", "observed",
        "z", "bh_q", "unknown_stage_share",
    },
    "model_coefs": {"run_id", "tree_step", "feature", "beta", "se", "ci_low", "ci_high"},
}


def _columns(engine, table):
    return {c["name"]: c for c in inspect(engine).get_columns(table)}


def test_create_all_creates_every_spec_table(engine):
    assert set(EXPECTED_COLUMNS) <= set(inspect(engine).get_table_names())


@pytest.mark.parametrize("table", sorted(EXPECTED_COLUMNS))
def test_table_has_key_columns(engine, table):
    assert EXPECTED_COLUMNS[table] <= set(_columns(engine, table))


@pytest.mark.parametrize("column", ["ev_loss", "stage", "stage_source"])
def test_decision_optional_columns_are_nullable(engine, column):
    assert _columns(engine, "decisions")[column]["nullable"] is True


def test_decision_actor_is_required(engine):
    assert _columns(engine, "decisions")["actor"]["nullable"] is False


def _hand(session):
    hand = Hand(source="pokerstars", raw_text="raw", tournament_id="T1", level=1, ante=0.0)
    session.add(hand)
    session.flush()
    return hand


def _decision(hand, **overrides):
    fields = dict(
        hand_id=hand.id, actor="hero", position="BTN", hole_cards="AhKh", hand_class="AKs",
        action="raise", size_bb=2.2, size_pot=None, eff_stack_bb=40.0, line="CO:fold",
    )
    fields.update(overrides)
    return Decision(**fields)


def test_decision_without_actor_is_rejected(session):
    session.add(_decision(_hand(session), actor=None))
    with pytest.raises(IntegrityError):
        session.flush()


def test_decision_without_stage_and_ev_loss_is_stored(session):
    session.add(_decision(_hand(session)))
    session.flush()


def test_decision_requires_existing_hand(session):
    session.add(Decision(
        hand_id=999, actor="hero", position="BTN", hole_cards="AhKh", hand_class="AKs",
        action="raise", eff_stack_bb=40.0, line="",
    ))
    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize("stage_source", ["proxy", "summary", "manual"])
def test_decision_accepts_known_stage_source(session, stage_source):
    session.add(_decision(_hand(session), stage="bubble", stage_source=stage_source))
    session.flush()


def test_decision_rejects_unknown_stage(session):
    session.add(_decision(_hand(session), stage="late"))
    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize("model", ["chipev", "icm:default", "icm:ft-9"])
def test_chart_set_accepts_known_models(session, model):
    session.add(ChartSet(
        model=model, ante="bb:1", source="synthetic-test", version="1", eligible_for_analysis=False
    ))
    session.flush()


@pytest.mark.parametrize("model", ["icm", "icm:", "cev", "ICM:x", ""])
def test_chart_set_rejects_unknown_models(session, model):
    session.add(ChartSet(
        model=model, ante="bb:1", source="synthetic-test", version="1", eligible_for_analysis=False
    ))
    with pytest.raises(IntegrityError):
        session.flush()
