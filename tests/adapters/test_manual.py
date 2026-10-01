import copy
import dataclasses
import json
from pathlib import Path

import pytest

from core.adapters.manual import ManualHandError, parse_manual, parse_manual_text
from core.adapters.pokerstars import parse_hand
from core.model import ParsedHand

FIXTURES = Path(__file__).parent / "fixtures"


def manual(name):
    return parse_manual_text((FIXTURES / name).read_text(encoding="utf-8"))


def pokerstars(name):
    return parse_hand((FIXTURES / name).read_text(encoding="utf-8"), hero_name="Hero")


def without_stage(hand):
    return dataclasses.replace(hand, stage=None, stage_source=None)


def rfi_data():
    return json.loads((FIXTURES / "rfi_each_ante.json").read_text(encoding="utf-8"))


def errors_of(data):
    with pytest.raises(ManualHandError) as info:
        parse_manual(data)
    return "\n".join(info.value.errors)


# --- эквивалентность с PokerStars --------------------------------------------


@pytest.mark.parametrize(
    "manual_file, hh_file",
    [("rfi_each_ante.json", "rfi_each_ante.txt"), ("threebet_jam_bb_ante.yaml", "threebet_jam_bb_ante.txt")],
)
def test_manual_hand_equals_pokerstars_hand(manual_file, hh_file):
    parsed = manual(manual_file)
    assert isinstance(parsed, ParsedHand)
    assert without_stage(parsed.hand) == pokerstars(hh_file)


def test_stage_comes_from_manual_input():
    assert (manual("rfi_each_ante.json").hand.stage, manual("rfi_each_ante.json").hand.stage_source) == ("mid", "manual")
    assert manual("threebet_jam_bb_ante.yaml").hand.stage == "bubble"


def test_raw_text_is_canonical_json_that_parses_back():
    parsed = manual("threebet_jam_bb_ante.yaml")
    assert json.loads(parsed.raw_text)["stage"] == "bubble"
    assert parse_manual_text(parsed.raw_text) == parsed


def test_optional_fields():
    data = rfi_data()
    for key in ("id", "tournament", "level"):
        del data[key]
    hand = parse_manual(data).hand
    assert (hand.external_id, hand.tournament_id, hand.level) == (None, None, None)


# --- стадия обязательна ------------------------------------------------------


def test_stage_is_required():
    data = rfi_data()
    del data["stage"]
    assert "stage" in errors_of(data)


@pytest.mark.parametrize("stage", ["late", "", None, 3])
def test_stage_must_be_known(stage):
    data = rfi_data()
    data["stage"] = stage
    assert "stage" in errors_of(data)


# --- ошибки ввода ------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.pop("blinds"), "blinds"),
        (lambda d: d.update(ante={"type": "straddle", "amount": 1}), "ante"),
        (lambda d: d.update(ante={"type": "each"}), "ante"),
        (lambda d: d.update(button=4.5), "button"),
        (lambda d: d.update(hero="Nobody"), "Nobody"),
        (lambda d: d.update(cards="AhAh"), "cards"),
        (lambda d: d["seats"].append({"seat": 1, "player": "dup", "stack": 100}), "seat 1"),
        (lambda d: d["seats"].append({"seat": 9, "player": "alpha", "stack": 100}), "alpha"),
        (lambda d: d["seats"][0].update(stack=-5), "stack"),
        (lambda d: d["actions"].append({"player": "ghost", "action": "fold"}), "ghost"),
        (lambda d: d["actions"].__setitem__(5, {"player": "Hero", "action": "raise"}), "to"),
        (lambda d: d["actions"].__setitem__(5, {"player": "Hero", "action": "limp"}), "limp"),
    ],
)
def test_invalid_input_is_rejected(mutate, message):
    data = rfi_data()
    mutate(data)
    assert message in errors_of(data)


def test_inconsistent_hand_is_rejected_by_normalization():
    data = rfi_data()
    data["actions"][5] = {"player": "Hero", "action": "raise", "to": 99999}
    assert "exceeds stack" in errors_of(data)


def test_all_structural_errors_are_reported_at_once():
    data = rfi_data()
    del data["stage"]
    data["cards"] = "XX"
    message = errors_of(data)
    assert "stage" in message and "cards" in message


@pytest.mark.parametrize("text", ["{not json", "- just\n- a list", ""])
def test_unparseable_text_is_rejected(text):
    with pytest.raises(ManualHandError):
        parse_manual_text(text)


def test_input_is_not_mutated():
    data = rfi_data()
    snapshot = copy.deepcopy(data)
    parse_manual(data)
    assert data == snapshot
