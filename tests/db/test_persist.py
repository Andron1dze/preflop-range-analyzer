import pytest
from sqlalchemy import select

from core.db.models import Decision, Hand
from core.db.persist import save_hand
from tests.factories import folds, make_hand


def test_save_hand_stores_raw_text_and_every_decision(session):
    hand = make_hand(folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 220)] + folds("p7", "p8"))
    stored = save_hand(session, hand, source="pokerstars", raw_text="RAW HH")
    session.commit()

    row = session.get(Hand, stored.id)
    assert row.raw_text == "RAW HH"
    assert row.external_id == "1"
    assert row.ante == pytest.approx(1.0)  # анте в bb

    decisions = session.scalars(select(Decision).order_by(Decision.id)).all()
    assert len(decisions) == 8


def test_hero_decision_carries_cards_and_hand_class(session):
    hand = make_hand(folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 220)] + folds("p7", "p8"))
    save_hand(session, hand, source="pokerstars", raw_text="RAW")
    session.flush()

    hero = session.scalars(select(Decision).where(Decision.actor == "hero")).one()
    assert hero.position == "BTN"
    assert hero.hole_cards == "AhKh"
    assert hero.hand_class == "AKs"
    assert hero.action == "raise"
    assert hero.size_bb == pytest.approx(2.2)
    assert hero.eff_stack_bb == pytest.approx(40.0)
    assert hero.line == "UTG:F,UTG1:F,LJ:F,HJ:F,CO:F"


def test_opponents_are_stored_by_name_without_cards(session):
    hand = make_hand(folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 220)] + folds("p7", "p8"))
    save_hand(session, hand, source="pokerstars", raw_text="RAW")
    session.flush()

    bb = session.scalars(select(Decision).where(Decision.actor == "p8")).one()
    assert bb.hole_cards is None and bb.hand_class is None


def test_stage_is_copied_to_decisions(session):
    hand = make_hand(folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 220)] + folds("p7", "p8"))
    hand.stage, hand.stage_source = "bubble", "manual"
    save_hand(session, hand, source="manual", raw_text="{}")
    session.flush()

    stages = set(session.execute(select(Decision.stage, Decision.stage_source)).all())
    assert stages == {("bubble", "manual")}


def test_all_in_flag_is_stored(session):
    hand = make_hand(
        folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 4000), ("p7", "fold", None), ("p8", "call", None)],
        stacks={"p8": 1500},
    )
    save_hand(session, hand, source="pokerstars", raw_text="RAW")
    session.flush()

    flags = dict(session.execute(select(Decision.actor, Decision.all_in)).all())
    assert flags == {"p1": False, "p2": False, "p3": False, "p4": False, "p5": False,
                     "hero": True, "p7": False, "p8": True}
