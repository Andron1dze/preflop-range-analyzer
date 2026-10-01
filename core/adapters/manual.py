"""Ручной ввод раздачи (JSON или YAML) → внутренняя модель раздачи.

Суммы — в фишках, как в живой игре; нормализация в bb — общая с PokerStars.

    id, tournament, level            — необязательные
    blinds: {sb, bb}
    ante: {type: none|each|bb, amount}
    button: <номер места>
    seats: [{seat, player, stack}]
    hero, cards
    actions: [{player, action: fold|check|call|raise, to}]   # to — только для raise
    stage: early|mid|bubble|itm|ft                           # обязательно

Сырьё для ре-парсинга — канонический JSON (сортированные ключи). Все ошибки
ввода собираются и выдаются разом; затем раздача проверяется нормализацией.
"""

import json
import math
from typing import Any

import yaml

from core.model import STAGES, Action, ActionKind, AnteType, Hand, ParsedHand, Seat
from core.normalize.hand_class import hand_class
from core.normalize.preflop import normalize


class ManualHandError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid hand:\n" + "\n".join(f"- {e}" for e in errors))


def parse_manual_text(text: str) -> ParsedHand:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ManualHandError([f"not valid JSON/YAML: {error}"]) from error
    return parse_manual(data)


def parse_manual(data: Any) -> ParsedHand:
    if not isinstance(data, dict):
        raise ManualHandError(["hand must be a mapping (JSON object)"])
    errors: list[str] = []

    blinds = data.get("blinds")
    if not isinstance(blinds, dict) or not all(_positive(blinds.get(k)) for k in ("sb", "bb")):
        errors.append("blinds: expected {sb, bb} with positive amounts")
        blinds = {"sb": 0, "bb": 1}

    ante_type, ante = _parse_ante(data.get("ante"), errors)

    button = data.get("button")
    if not _is_int(button):
        errors.append(f"button: expected a seat number, got {button!r}")

    seats = _parse_seats(data.get("seats"), errors)
    players = {s.player for s in seats}

    hero = data.get("hero")
    if hero not in players:
        errors.append(f"hero {hero!r} is not seated")

    cards = data.get("cards")
    try:
        cards = str(cards).replace(" ", "")
        hand_class(cards)
    except ValueError:
        errors.append(f"cards: {data.get('cards')!r} are not two valid cards")

    actions = _parse_actions(data.get("actions"), players, errors)

    stage = data.get("stage")
    if stage not in STAGES:
        errors.append(f"stage: required, one of {', '.join(STAGES)}; got {stage!r}")

    level = data.get("level")
    if level is not None and not _is_int(level):
        errors.append(f"level: expected an integer, got {level!r}")

    if errors:
        raise ManualHandError(errors)

    hand = Hand(
        external_id=_optional_str(data.get("id")),
        tournament_id=_optional_str(data.get("tournament")),
        level=level,
        small_blind=float(blinds["sb"]),
        big_blind=float(blinds["bb"]),
        ante=ante,
        ante_type=ante_type,
        button_seat=button,
        seats=seats,
        hero=hero,
        hero_cards=cards,
        actions=actions,
        stage=stage,
        stage_source="manual",
    )
    try:
        normalize(hand)
    except ValueError as error:
        raise ManualHandError([f"actions: {error}"]) from error
    return ParsedHand(hand, json.dumps(data, sort_keys=True, ensure_ascii=False))


def _parse_ante(raw: Any, errors: list[str]) -> tuple[AnteType, float]:
    if raw is None:
        return AnteType.NONE, 0.0
    if not isinstance(raw, dict) or raw.get("type") not in {t.value for t in AnteType}:
        errors.append("ante: expected {type: none|each|bb, amount}")
        return AnteType.NONE, 0.0
    ante_type = AnteType(raw["type"])
    amount = raw.get("amount", 0 if ante_type is AnteType.NONE else None)
    if not _number(amount) or amount < 0:
        errors.append(f"ante: amount must be a non-negative number, got {amount!r}")
        return ante_type, 0.0
    return ante_type, float(amount)


def _parse_seats(raw: Any, errors: list[str]) -> list[Seat]:
    if not isinstance(raw, list) or len(raw) < 2:
        errors.append("seats: at least two seats are required")
        return []
    seats, numbers, names = [], set(), set()
    for index, seat in enumerate(raw):
        if not isinstance(seat, dict):
            errors.append(f"seats[{index}]: expected {{seat, player, stack}}")
            continue
        number, player, stack = seat.get("seat"), seat.get("player"), seat.get("stack")
        if not _is_int(number):
            errors.append(f"seats[{index}]: seat number {number!r} is not an integer")
        elif number in numbers:
            errors.append(f"seat {number} is listed twice")
        if not isinstance(player, str) or not player:
            errors.append(f"seats[{index}]: player name is required")
        elif player in names:
            errors.append(f"player {player!r} is seated twice")
        if not _positive(stack):
            errors.append(f"seats[{index}]: stack {stack!r} must be a positive number")
        numbers.add(number)
        names.add(player)
        if _is_int(number) and isinstance(player, str) and _positive(stack):
            seats.append(Seat(number, player, float(stack)))
    return seats


def _parse_actions(raw: Any, players: set[str], errors: list[str]) -> list[Action]:
    if not isinstance(raw, list):
        errors.append("actions: expected a list")
        return []
    actions = []
    kinds = {k.value for k in ActionKind}
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            errors.append(f"action #{index}: expected {{player, action, to}}")
            continue
        player, kind, to = item.get("player"), item.get("action"), item.get("to")
        if player not in players:
            errors.append(f"action #{index}: player {player!r} is not seated")
        if kind not in kinds:
            errors.append(f"action #{index}: unknown action {kind!r} (fold, check, call, raise)")
            continue
        if kind == ActionKind.RAISE and not _positive(to):
            errors.append(f"action #{index}: raise needs a positive 'to' amount")
            continue
        actions.append(Action(player, ActionKind(kind), float(to) if kind == ActionKind.RAISE else None))
    return actions


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value: Any) -> bool:
    return _number(value) and value > 0


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
