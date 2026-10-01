from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db import models
from core.model import Hand, ParsedHand
from core.normalize.hand_class import hand_class
from core.normalize.preflop import normalize


def save_hand(session: Session, hand: Hand, *, source: str, raw_text: str) -> models.Hand:
    """Сохраняет раздачу и все её префлоп-решения. Коммит — на вызывающем."""
    row = models.Hand(
        source=source,
        external_id=hand.external_id,
        raw_text=raw_text,
        tournament_id=hand.tournament_id,
        level=hand.level,
        ante=hand.ante / hand.big_blind,
        ante_type=hand.ante_type.value,
    )
    session.add(row)
    session.flush()

    hero_class = hand_class(hand.hero_cards) if hand.hero_cards else None
    for point in normalize(hand):
        is_hero = point.player == hand.hero
        session.add(
            models.Decision(
                hand_id=row.id,
                actor="hero" if is_hero else point.player,
                position=point.position,
                hole_cards=hand.hero_cards if is_hero else None,
                hand_class=hero_class if is_hero else None,
                action=point.kind.value,
                size_bb=point.size_bb,
                size_pot=point.size_pot,
                eff_stack_bb=point.eff_stack_bb,
                all_in=point.all_in,
                line=point.line,
                stage=hand.stage,
                stage_source=hand.stage_source,
            )
        )
    return row


@dataclass
class ImportReport:
    imported: int = 0
    duplicates: int = 0
    # (external_id, сообщение) раздач, которые не прошли нормализацию.
    errors: list[tuple[str | None, str]] = field(default_factory=list)


def import_hands(session: Session, parsed: Iterable[ParsedHand], *, source: str) -> ImportReport:
    """Сохраняет разобранные раздачи, пропуская уже импортированные. Коммит — на вызывающем."""
    parsed = list(parsed)
    ids = {p.hand.external_id for p in parsed if p.hand.external_id is not None}
    seen = set(
        session.scalars(
            select(models.Hand.external_id).where(
                models.Hand.source == source, models.Hand.external_id.in_(ids)
            )
        )
    )

    report = ImportReport()
    for item in parsed:
        external_id = item.hand.external_id
        if external_id is not None and external_id in seen:
            report.duplicates += 1
            continue
        try:
            # Savepoint: раздача, упавшая на нормализации, не оставляет частичных строк.
            with session.begin_nested():
                save_hand(session, item.hand, source=source, raw_text=item.raw_text)
        except ValueError as error:
            report.errors.append((external_id, str(error)))
            continue
        seen.add(external_id)
        report.imported += 1
    return report
