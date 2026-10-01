"""Маппинг решений на узлы charts/40bb_core.yaml (блайнды 50/100, анте 12.5 с каждого, стеки 40bb)."""

from pathlib import Path

import pytest
from sqlalchemy import func, select

from core.analysis.config import AnalysisConfig
from core.analysis.pipeline import run_analysis
from core.charts.format import load_chart_file
from core.charts.store import save_chart_set
from core.db.models import Decision, DecisionNodeMap, Node, NodeStat
from core.db.persist import save_hand
from core.mapping.distance import MappingConfig
from core.mapping.remap import remap
from core.model import AnteType
from tests.factories import folds, make_hand

CHART = Path(__file__).parents[2] / "charts" / "40bb_core.yaml"


@pytest.fixture
def chart_set(session):
    row = save_chart_set(session, load_chart_file(CHART))
    session.flush()
    return row


def add(session, actions, external_id, **kwargs):
    hand = make_hand(actions, ante=12.5, ante_type=AnteType.EACH, **kwargs)
    hand.external_id = external_id
    save_hand(session, hand, source="pokerstars", raw_text="RAW")
    session.flush()


def mapping_of(session, version, actor_position):
    actor, position = actor_position
    return session.execute(
        select(Node.key, DecisionNodeMap.distance, DecisionNodeMap.weight)
        .join(DecisionNodeMap, DecisionNodeMap.node_id == Node.id)
        .join(Decision, Decision.id == DecisionNodeMap.decision_id)
        .where(DecisionNodeMap.mapping_version == version, Decision.actor == actor, Decision.position == position)
    ).all()


def test_btn_open_maps_to_rfi_node_with_full_weight(session, chart_set):
    # Эффективный стек после анте 39.875bb; с анте обратно — ровно 40bb.
    add(session, folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 200)] + folds("p7", "p8"), "1")
    report = remap(session, chart_set.id, MappingConfig())
    [(key, distance, weight)] = mapping_of(session, report.version, ("hero", "BTN"))
    assert key == "40bb|HERO=BTN"
    assert distance == pytest.approx(0.0, abs=1e-9)
    assert weight == pytest.approx(1.0)


def test_bb_against_bigger_open_gets_reduced_weight(session, chart_set):
    add(
        session,
        folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 250), ("p7", "fold", None), ("p8", "call", None)],
        "1",
        hero="p8",
    )
    report = remap(session, chart_set.id, MappingConfig())
    [(key, distance, weight)] = mapping_of(session, report.version, ("hero", "BB"))
    assert key == "40bb|BTN:open|HERO=BB"
    assert distance == pytest.approx(1.0, abs=1e-6)
    assert weight == pytest.approx(0.61, abs=0.005)


def test_far_decision_is_stored_with_zero_weight(session, chart_set):
    add(
        session,
        folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 350), ("p7", "fold", None), ("p8", "fold", None)],
        "1",
        hero="p8",
    )
    report = remap(session, chart_set.id, MappingConfig())
    [(_, distance, weight)] = mapping_of(session, report.version, ("hero", "BB"))
    assert distance == pytest.approx(2.51, abs=0.005)
    assert weight == 0.0
    assert report.far == 1


def test_decisions_without_node_are_reported_not_stored(session, chart_set):
    # UTG1 после открытия UTG: узла "UTG:open|HERO=UTG1" в чартах нет.
    add(session, [("p1", "raise", 200)] + folds("p2", "p3", "p4", "p5", "p6", "p7", "p8"), "1")
    report = remap(session, chart_set.id, MappingConfig())
    stored = session.scalar(select(func.count()).select_from(DecisionNodeMap))
    assert stored == report.mapped + report.far
    assert report.mapped == 1  # только UTG — открытие
    assert report.unmapped == 7


def test_remap_is_idempotent_and_versioned(session, chart_set):
    add(session, folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 200)] + folds("p7", "p8"), "1")
    first = remap(session, chart_set.id, MappingConfig())
    second = remap(session, chart_set.id, MappingConfig())
    assert first.version == second.version
    assert session.scalar(select(func.count()).select_from(DecisionNodeMap)) == first.mapped + first.far

    other = remap(session, chart_set.id, MappingConfig(cutoff=0.5))
    assert other.version != first.version
    versions = session.scalars(select(DecisionNodeMap.mapping_version).distinct()).all()
    assert sorted(versions) == sorted([first.version, other.version])


def test_mapped_decisions_flow_into_analysis(session, chart_set):
    for i in range(40):
        add(session, folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 200)] + folds("p7", "p8"), str(i))
    report = remap(session, chart_set.id, MappingConfig())
    run = run_analysis(
        session, chart_set_id=chart_set.id, mapping_version=report.version, config=AnalysisConfig(), allow_synthetic=True,
    )
    session.flush()
    actions = session.scalars(
        select(NodeStat.action).where(NodeStat.run_id == run.id, NodeStat.zone.is_(None), NodeStat.hand_group.is_(None))
    ).all()
    assert sorted(actions) == ["fold", "raise"]
