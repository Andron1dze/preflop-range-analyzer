"""Математика нормализации. Блайнды 50/100, BB-ante 100, стеки 4000 (40bb), если не указано иное."""

import pytest

from core.model import AnteType
from core.normalize.preflop import normalize
from tests.factories import folds, make_hand


def by_player(points):
    return {p.player: p for p in points}


def test_every_preflop_action_becomes_a_decision_point():
    hand = make_hand(folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 220)] + folds("p7", "p8"))
    points = normalize(hand)
    assert [p.player for p in points] == ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
    assert [p.position for p in points] == ["UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]


def test_open_raise_in_bb_and_pot_fraction():
    # Пот до решения: SB 50 + BB 100 + BB-ante 100 = 250 (2.5bb).
    # Пот-сайз рейз (как у солверов): прибавка сверх колла / пот после колла = (220-100)/(250+100).
    hand = make_hand(folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 220)] + folds("p7", "p8"))
    btn = by_player(normalize(hand))["p6"]
    assert btn.kind == "raise"
    assert btn.size_bb == pytest.approx(2.2)
    assert btn.size_pot == pytest.approx(120 / 350)
    assert btn.pot_bb == pytest.approx(2.5)
    assert btn.all_in is False


def test_three_bet_pot_fraction():
    # CO открывает до 220, BTN 3-бетит до 700.
    # Пот перед BTN: 250 + 220 = 470; колл BTN = 220; прибавка = 700 - 220 = 480.
    hand = make_hand(
        folds("p1", "p2", "p3", "p4") + [("p5", "raise", 220), ("p6", "raise", 700)] + folds("p7", "p8", "p5")
    )
    btn = by_player(normalize(hand))["p6"]
    assert btn.size_bb == pytest.approx(7.0)
    assert btn.pot_bb == pytest.approx(4.7)
    assert btn.size_pot == pytest.approx(480 / (470 + 220))


def test_call_size_is_amount_to_and_has_no_pot_fraction():
    hand = make_hand(
        folds("p1", "p2", "p3", "p4") + [("p5", "raise", 220), ("p6", "call", None)] + folds("p7", "p8")
    )
    btn = by_player(normalize(hand))["p6"]
    assert btn.kind == "call"
    assert btn.size_bb == pytest.approx(2.2)
    assert btn.size_pot is None


def test_fold_and_check_have_no_size():
    hand = make_hand(folds("p1", "p2", "p3", "p4", "p5", "p6") + [("p7", "call", None), ("p8", "check", None)])
    points = by_player(normalize(hand))
    assert points["p6"].size_bb is None and points["p6"].size_pot is None
    assert points["p8"].kind == "check" and points["p8"].size_bb is None
    # Комплит SB — это колл до 1bb.
    assert points["p7"].size_bb == pytest.approx(1.0)


def test_effective_stack_is_capped_by_biggest_live_opponent():
    stacks = {"p6": 4000, "p7": 6000, "p8": 2500}
    hand = make_hand(folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 220)] + folds("p7", "p8"), stacks=stacks)
    # Живы SB (6000) и BB: герою ограничивает только собственный стек.
    assert by_player(normalize(hand))["p6"].eff_stack_bb == pytest.approx(40.0)


def test_effective_stack_subtracts_bb_ante_from_big_blind():
    stacks = {"p7": 6000, "p8": 2500}
    hand = make_hand(
        folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 220), ("p7", "fold", None), ("p8", "call", None)],
        stacks=stacks,
    )
    points = by_player(normalize(hand))
    # BB решает против BTN: min(2500 - 100, 4000) = 2400 → 24bb.
    assert points["p8"].eff_stack_bb == pytest.approx(24.0)


def test_effective_stack_with_ante_paid_by_everyone():
    hand = make_hand(
        folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 220)] + folds("p7", "p8"),
        ante=12.0,
        ante_type=AnteType.EACH,
    )
    btn = by_player(normalize(hand))["p6"]
    assert btn.eff_stack_bb == pytest.approx((4000 - 12) / 100)
    # Пот: блайнды 150 + анте 8 × 12 = 246.
    assert btn.pot_bb == pytest.approx(2.46)


def test_no_ante():
    hand = make_hand(
        folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 250)] + folds("p7", "p8"),
        ante=0.0,
        ante_type=AnteType.NONE,
    )
    btn = by_player(normalize(hand))["p6"]
    assert btn.eff_stack_bb == pytest.approx(40.0)
    assert btn.pot_bb == pytest.approx(1.5)


def test_jam_and_short_all_in_call():
    stacks = {"p8": 1500}
    hand = make_hand(
        folds("p1", "p2", "p3", "p4", "p5") + [("p6", "raise", 4000), ("p7", "fold", None), ("p8", "call", None)],
        stacks=stacks,
    )
    points = by_player(normalize(hand))
    assert points["p6"].all_in is True
    assert points["p6"].size_bb == pytest.approx(40.0)
    # BB коллирует на всё, что осталось после анте: 1500 - 100 = 1400.
    assert points["p8"].all_in is True
    assert points["p8"].size_bb == pytest.approx(14.0)


def test_line_records_previous_actions_by_position():
    hand = make_hand(
        folds("p1", "p2", "p3", "p4") + [("p5", "raise", 220), ("p6", "raise", 700)] + folds("p7", "p8")
        + [("p5", "raise", 4000), ("p6", "call", None)]
    )
    points = normalize(hand)
    assert points[0].line == ""
    btn_first = points[5]
    assert btn_first.line == "UTG:F,UTG1:F,LJ:F,HJ:F,CO:R2.2"
    btn_second = points[-1]
    assert btn_second.line == "UTG:F,UTG1:F,LJ:F,HJ:F,CO:R2.2,BTN:R7,SB:F,BB:F,CO:R40ai"


@pytest.mark.parametrize(
    "actions, message",
    [
        ([("p9", "fold", None)], "p9"),
        (folds("p1", "p1"), "p1"),
        ([("p1", "check", None)], "check"),
        ([("p1", "raise", 100)], "raise"),
        ([("p1", "raise", 5000)], "stack"),
    ],
)
def test_invalid_actions_are_rejected(actions, message):
    with pytest.raises(ValueError, match=message):
        normalize(make_hand(actions))
