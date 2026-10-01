import pytest
from sqlalchemy import func, select

from core.charts.format import SYNTHETIC_SOURCE, parse_chart_set
from core.charts.store import save_chart_set
from core.db.models import ChartSet, Node, NodeStrategy
from tests.charts.test_format import chart_data


def test_chart_set_is_stored_with_nodes_and_strategies(session):
    stored = save_chart_set(session, parse_chart_set(chart_data(source="gto-wizard-export-2026-09")))
    session.commit()

    row = session.get(ChartSet, stored.id)
    assert (row.model, row.ante, row.version) == ("chipev", "bb:1", "1")
    assert row.eligible_for_analysis is True

    node = session.scalars(select(Node)).one()
    assert node.key == "40bb|HERO=BTN"
    assert node.sizings == {"BTN": 2.2}
    assert node.stack_bucket == "40bb"

    aa = session.scalars(select(NodeStrategy).where(NodeStrategy.hand_class == "AA")).all()
    assert [(s.action, s.freq) for s in aa] == [("raise", 1.0)]
    # Нулевые частоты не храним: 168 рук × 2 действия + AA × 1.
    assert session.scalar(select(func.count()).select_from(NodeStrategy)) == 168 * 2 + 1


def test_synthetic_chart_set_is_stored_as_not_eligible(session):
    stored = save_chart_set(session, parse_chart_set(chart_data(source=SYNTHETIC_SOURCE)))
    session.flush()
    assert stored.eligible_for_analysis is False


def test_same_source_and_version_cannot_be_loaded_twice(session):
    chart = parse_chart_set(chart_data())
    save_chart_set(session, chart)
    session.flush()
    with pytest.raises(ValueError, match="already loaded"):
        save_chart_set(session, chart)
