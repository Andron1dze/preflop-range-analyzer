"""Текстовые HH PokerStars (турниры, NLHE) → внутренняя модель раздачи.

Разбирается только префлоп: всё после первой строки "*** ..." вслед за
"*** HOLE CARDS ***" (флоп, шоудаун, итоги) игнорируется.
"""

import re
from dataclasses import dataclass, field

from core.model import Action, ActionKind, AnteType, Hand, ParsedHand, Seat


class PokerStarsParseError(ValueError):
    pass


@dataclass
class HandParseError:
    external_id: str | None
    message: str


@dataclass
class ParseResult:
    hands: list[ParsedHand] = field(default_factory=list)
    errors: list[HandParseError] = field(default_factory=list)


_HAND_START = "PokerStars Hand #"
_HEADER_RE = re.compile(
    r"^PokerStars Hand #(?P<hand>\d+): Tournament #(?P<tournament>\d+), "
    r".*Hold'em No Limit - Level (?P<level>[IVXLCDM]+) "
    r"\((?P<sb>[\d.]+)/(?P<bb>[\d.]+)\)"
)
_HAND_ID_RE = re.compile(r"^PokerStars Hand #(\d+)")
_TABLE_RE = re.compile(r"^Table '.*' \d+-max Seat #(?P<button>\d+) is the button")
_SEAT_RE = re.compile(r"^Seat (?P<number>\d+): (?P<name>.+) \((?P<stack>[\d.]+) in chips\)(?P<rest>.*)$")
_DEALT_RE = re.compile(r"^Dealt to (?P<name>.+) \[(?P<cards>\w\w \w\w)\]$")

_ALL_IN = r"(?: and is all-in)?$"
_POST_RES = {
    "ante": re.compile(rf"^posts the ante (?P<amount>[\d.]+){_ALL_IN}"),
    "small": re.compile(rf"^posts small blind (?P<amount>[\d.]+){_ALL_IN}"),
    "big": re.compile(rf"^posts big blind (?P<amount>[\d.]+){_ALL_IN}"),
}
_FOLD_RE = re.compile(r"^folds(?: \[.*\])?$")
_CALL_RE = re.compile(rf"^calls [\d.]+{_ALL_IN}")
_RAISE_RE = re.compile(rf"^raises [\d.]+ to (?P<to>[\d.]+){_ALL_IN}")
# Строки игрока, не являющиеся решением на префлопе.
_IGNORED_PLAYER_RE = re.compile(r"^(doesn't show hand|mucks hand|shows \[.*)$")

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def parse_file(text: str, hero_name: str) -> ParseResult:
    """Разбивает файл на раздачи; битая раздача попадает в errors, остальные разбираются."""
    result = ParseResult()
    for raw in split_hands(text):
        try:
            result.hands.append(ParsedHand(parse_hand(raw, hero_name), raw))
        except PokerStarsParseError as error:
            match = _HAND_ID_RE.match(raw)
            result.errors.append(HandParseError(match and match.group(1), str(error)))
    return result


def split_hands(text: str) -> list[str]:
    text = text.lstrip("﻿").replace("\r\n", "\n")
    starts = [m.start() for m in re.finditer(rf"^{re.escape(_HAND_START)}", text, re.MULTILINE)]
    return [text[a:b].strip() for a, b in zip(starts, [*starts[1:], len(text)])]


def parse_hand(raw: str, hero_name: str) -> Hand:
    lines = [line.strip() for line in raw.strip().replace("\r\n", "\n").split("\n")]
    header = _HEADER_RE.match(lines[0])
    if not header:
        raise PokerStarsParseError(f"unsupported header: {lines[0]!r}")
    hand_id = header["hand"]

    def fail(message: str) -> PokerStarsParseError:
        return PokerStarsParseError(f"hand #{hand_id}: {message}")

    table = _TABLE_RE.match(lines[1]) if len(lines) > 1 else None
    if not table:
        raise fail(f"no button line: {lines[1] if len(lines) > 1 else ''!r}")

    seats: list[Seat] = []
    antes: dict[str, float] = {}
    big_blind_player = None
    hero_cards = None
    actions: list[Action] = []
    section = "setup"

    for line in lines[2:]:
        if line == "*** HOLE CARDS ***":
            section = "preflop"
            continue
        if line.startswith("*** "):
            if section == "preflop":
                break
            continue

        if section == "setup":
            seat = _SEAT_RE.match(line)
            if seat:
                if "out of hand" not in seat["rest"]:
                    seats.append(Seat(int(seat["number"]), seat["name"], float(seat["stack"])))
                continue
            player, rest = _split_player(line, seats)
            if player is None:
                continue
            for kind, pattern in _POST_RES.items():
                post = pattern.match(rest)
                if post:
                    if kind == "ante":
                        antes[player] = float(post["amount"])
                    elif kind == "big":
                        big_blind_player = player
                    break
            else:
                raise fail(f"unrecognized line: {line!r}")
            continue

        dealt = _DEALT_RE.match(line)
        if dealt:
            if dealt["name"] == hero_name:
                hero_cards = dealt["cards"].replace(" ", "")
            continue
        player, rest = _split_player(line, seats)
        if player is None or _IGNORED_PLAYER_RE.match(rest):
            continue
        actions.append(_parse_action(player, rest, line, fail))

    names = {s.player for s in seats}
    if hero_name not in names:
        raise fail(f"hero {hero_name!r} is not seated")
    if hero_cards is None:
        raise fail(f"hero cards not found for {hero_name!r}")

    return Hand(
        external_id=hand_id,
        tournament_id=header["tournament"],
        level=_roman_to_int(header["level"]),
        small_blind=float(header["sb"]),
        big_blind=float(header["bb"]),
        ante=max(antes.values(), default=0.0),
        ante_type=_ante_type(antes, big_blind_player),
        button_seat=int(table["button"]),
        seats=seats,
        hero=hero_name,
        hero_cards=hero_cards,
        actions=actions,
    )


def _split_player(line: str, seats: list[Seat]) -> tuple[str | None, str]:
    """Строка "<игрок>: <действие>" → (игрок, действие). Ники могут содержать пробелы и ':'."""
    candidates = [s.player for s in seats if line.startswith(s.player + ": ")]
    if not candidates:
        return None, line
    player = max(candidates, key=len)
    return player, line[len(player) + 2:]


def _parse_action(player: str, rest: str, line: str, fail) -> Action:
    if _FOLD_RE.match(rest):
        return Action(player, ActionKind.FOLD)
    if rest == "checks":
        return Action(player, ActionKind.CHECK)
    if _CALL_RE.match(rest):
        return Action(player, ActionKind.CALL)
    raise_ = _RAISE_RE.match(rest)
    if raise_:
        return Action(player, ActionKind.RAISE, float(raise_["to"]))
    raise fail(f"unrecognized line: {line!r}")


def _ante_type(antes: dict[str, float], big_blind_player: str | None) -> AnteType:
    if not antes:
        return AnteType.NONE
    if list(antes) == [big_blind_player]:
        return AnteType.BB
    return AnteType.EACH


def _roman_to_int(numeral: str) -> int:
    values = [_ROMAN[c] for c in numeral]
    return sum(-v if v < nxt else v for v, nxt in zip(values, [*values[1:], 0]))
