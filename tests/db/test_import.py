from pathlib import Path

from sqlalchemy import func, select

from core.adapters.pokerstars import parse_file
from core.db.models import Decision, Hand
from core.db.persist import import_hands
from core.model import ParsedHand
from tests.factories import folds, make_hand

FIXTURES = Path(__file__).parents[1] / "adapters" / "fixtures"
TEXT = "\n\n".join(
    (FIXTURES / name).read_text(encoding="utf-8")
    for name in ("rfi_each_ante.txt", "threebet_jam_bb_ante.txt")
)


def test_import_stores_hands_with_raw_text(session):
    report = import_hands(session, parse_file(TEXT, hero_name="Hero").hands, source="pokerstars")
    session.commit()

    assert (report.imported, report.duplicates) == (2, 0)
    raw = session.scalars(select(Hand.raw_text).order_by(Hand.external_id)).all()
    assert raw[0].startswith("PokerStars Hand #250000000001")
    hero_actions = session.scalars(
        select(Decision.action).where(Decision.actor == "hero").order_by(Decision.id)
    ).all()
    assert hero_actions == ["raise", "raise", "call"]


def test_reimport_does_not_duplicate(session):
    parsed = parse_file(TEXT, hero_name="Hero").hands
    import_hands(session, parsed, source="pokerstars")
    session.commit()

    report = import_hands(session, parsed + parsed, source="pokerstars")
    session.commit()

    assert (report.imported, report.duplicates) == (0, 4)
    assert session.scalar(select(func.count()).select_from(Hand)) == 2


def test_hand_failing_normalization_leaves_no_rows_and_others_import(session):
    good = make_hand(folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 220)] + folds("p7", "p8"))
    # Рейз больше стека: разбор прошёл бы, нормализация — нет (после записи решений p1..p5).
    bad = make_hand(folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 9999)])
    bad.external_id = "2"

    report = import_hands(
        session, [ParsedHand(bad, "BAD"), ParsedHand(good, "GOOD")], source="pokerstars"
    )
    session.commit()

    assert report.imported == 1
    assert [external_id for external_id, _ in report.errors] == ["2"]
    assert session.scalars(select(Hand.raw_text)).all() == ["GOOD"]
    assert session.scalar(select(func.count()).select_from(Decision)) == 8
