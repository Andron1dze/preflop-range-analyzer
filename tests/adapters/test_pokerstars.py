from pathlib import Path

import pytest

from core.adapters.pokerstars import PokerStarsParseError, parse_file, parse_hand
from core.model import Action, ActionKind, AnteType, Seat
from core.normalize.preflop import normalize

FIXTURES = Path(__file__).parent / "fixtures"
RFI = (FIXTURES / "rfi_each_ante.txt").read_text(encoding="utf-8")
THREEBET_JAM = (FIXTURES / "threebet_jam_bb_ante.txt").read_text(encoding="utf-8")


def fold(player):
    return Action(player, ActionKind.FOLD)


def test_rfi_hand_header_seats_and_blinds():
    hand = parse_hand(RFI, hero_name="Hero")
    assert hand.external_id == "250000000001"
    assert hand.tournament_id == "3700000001"
    assert hand.level == 5
    assert (hand.small_blind, hand.big_blind) == (50, 100)
    assert (hand.ante, hand.ante_type) == (12, AnteType.EACH)
    assert hand.button_seat == 6
    assert hand.seats[3] == Seat(4, "Big Fish 77", 4000)
    assert len(hand.seats) == 8  # sitting out — всё равно в раздаче
    assert (hand.hero, hand.hero_cards) == ("Hero", "AhKh")


def test_rfi_hand_actions_skip_chat_and_post_action_lines():
    hand = parse_hand(RFI, hero_name="Hero")
    assert hand.actions == [
        fold("alpha"), fold("bravo"), fold("charlie"), fold("Big Fish 77"), fold("echo"),
        Action("Hero", ActionKind.RAISE, 220),
        fold("golf"), fold("hotel"),
    ]


def test_rfi_hand_normalizes():
    hero = next(p for p in normalize(parse_hand(RFI, hero_name="Hero")) if p.player == "Hero")
    assert hero.position == "BTN"
    assert hero.size_bb == pytest.approx(2.2)
    # Живы golf (6000 − 12) и hotel: эффективный стек ограничен героем, 4000 − 12.
    assert hero.eff_stack_bb == pytest.approx(39.88)
    assert hero.pot_bb == pytest.approx(2.46)
    assert hero.line == "UTG:F,UTG1:F,LJ:F,HJ:F,CO:F"


def test_threebet_jam_hand_with_bb_ante_and_empty_seat():
    hand = parse_hand(THREEBET_JAM, hero_name="Hero")
    assert hand.level == 8
    assert (hand.ante, hand.ante_type) == (200, AnteType.BB)
    assert [s.number for s in hand.seats] == [1, 2, 3, 5, 6, 7, 8]
    assert hand.hero_cards == "AcAd"
    assert hand.actions == [
        fold("alpha"), fold("bravo"), fold("charlie"),
        Action("Villain", ActionKind.RAISE, 440),
        Action("Hero", ActionKind.RAISE, 1300),
        fold("golf"), fold("hotel"),
        Action("Villain", ActionKind.RAISE, 8000),
        Action("Hero", ActionKind.CALL),
    ]


def test_threebet_jam_hand_normalizes():
    points = normalize(parse_hand(THREEBET_JAM, hero_name="Hero"))
    hero_3bet, hero_call = [p for p in points if p.player == "Hero"]
    jam = next(p for p in points if p.player == "Villain" and p.all_in)

    assert hero_3bet.size_bb == pytest.approx(6.5)
    assert hero_3bet.size_pot == pytest.approx(860 / (940 + 440))
    assert jam.size_bb == pytest.approx(40.0)
    assert hero_call.size_bb == pytest.approx(40.0)
    assert hero_call.eff_stack_bb == pytest.approx(40.0)
    assert hero_call.line == "UTG1:F,LJ:F,HJ:F,CO:R2.2,BTN:R6.5,SB:F,BB:F,CO:R40ai"


def test_raw_text_is_kept_per_hand():
    result = parse_file(RFI + "\n\n\n" + THREEBET_JAM, hero_name="Hero")
    assert [p.raw_text for p in result.hands] == [RFI.strip(), THREEBET_JAM.strip()]


def test_file_with_bom_and_crlf_is_split_into_hands():
    text = "﻿" + (RFI + "\n\n" + THREEBET_JAM).replace("\n", "\r\n")
    result = parse_file(text, hero_name="Hero")
    assert [p.hand.external_id for p in result.hands] == ["250000000001", "250000000002"]
    assert result.errors == []


def test_bad_hand_is_reported_and_the_rest_are_parsed():
    broken = RFI.replace("250000000001", "250000000099").replace("Hero: raises 120 to 220", "Hero: dances")
    result = parse_file("\n\n".join([RFI, broken, THREEBET_JAM]), hero_name="Hero")
    assert [p.hand.external_id for p in result.hands] == ["250000000001", "250000000002"]
    [error] = result.errors
    assert error.external_id == "250000000099"
    assert "Hero: dances" in error.message


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: t.replace("Dealt to Hero [Ah Kh]\n", ""), "cards"),
        (lambda t: t.replace("Hold'em No Limit", "Omaha Pot Limit"), "header"),
        (lambda t: t.replace("Seat #6 is the button", "Seat #4 is the bottom"), "button"),
    ],
)
def test_malformed_hand_raises(mutate, message):
    with pytest.raises(PokerStarsParseError, match=message):
        parse_hand(mutate(RFI), hero_name="Hero")


def test_hero_must_be_seated():
    with pytest.raises(PokerStarsParseError, match="Nobody"):
        parse_hand(RFI, hero_name="Nobody")
