from sqlalchemy.orm import Session

from core.db import models
from core.model import Hand
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
                line=point.line,
                stage=hand.stage,
                stage_source=hand.stage_source,
            )
        )
    return row
