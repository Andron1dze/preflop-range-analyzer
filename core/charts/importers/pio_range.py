"""Строки диапазонов Pio/GTO+ → канонический формат чарта.

Строка — токены через запятую, у каждого необязательный вес `:w` в [0, 1]:
    AA, AKs, AKo, AK (обе масти), AhKh (комбо),
    22+, A2s+, ATo+ (до старшей карты), 99-66, KTo-K8o (диапазоны).

Вес класса = средний вес по всем его комбо (6 у пар, 4 у одномастных,
12 у разномастных), поэтому `AhKh:0.5` даёт AKs = 0.125. Одно комбо,
указанное дважды, — ошибка.
"""

import re
from itertools import combinations, product
from typing import Any

from core.normalize.hand_class import ALL_HAND_CLASSES, RANKS, SUITS


class RangeParseError(ValueError):
    pass


_RANK = "[AKQJT98765432]"
_COMBO_RE = re.compile(rf"^({_RANK})([cdhs])({_RANK})([cdhs])$")
_CLASS_RE = re.compile(rf"^({_RANK})({_RANK})([so]?)$")

# Слабейшая карта сначала — удобно для "+" и диапазонов.
_ASCENDING = RANKS[::-1]


def parse_range(text: str) -> dict[str, float]:
    """Строка диапазона → вес каждого из 169 классов (отсутствующие — 0)."""
    combo_weights: dict[frozenset[str], float] = {}
    for token in (t.strip() for t in text.split(",")):
        if not token:
            continue
        body, weight = _split_weight(token)
        for combo in _expand(body, token):
            if combo in combo_weights:
                raise RangeParseError(f"{token}: combo {''.join(sorted(combo))} is specified more than once")
            combo_weights[combo] = weight

    totals = dict.fromkeys(ALL_HAND_CLASSES, 0.0)
    for combo, weight in combo_weights.items():
        totals[_class_of(combo)] += weight
    return {hand: total / len(_class_combos(hand)) for hand, total in totals.items()}


def strategy_from_ranges(
    ranges: dict[str, str], *, remainder: str = "fold"
) -> dict[str, dict[str, float]]:
    """{действие: строка диапазона} → {класс: {действие: частота}}; остаток до 1 — в remainder."""
    weights = {}
    for action, text in ranges.items():
        try:
            weights[action] = parse_range(text)
        except RangeParseError as error:
            raise RangeParseError(f"{action}: {error}") from error

    strategy = {}
    for hand in ALL_HAND_CLASSES:
        freqs = {action: w[hand] for action, w in weights.items() if w[hand] > 0}
        total = sum(freqs.values())
        if total > 1 + 1e-9:
            detail = ", ".join(f"{a} {f:g}" for a, f in freqs.items())
            raise RangeParseError(f"{hand}: actions sum to {total:g} > 1 ({detail})")
        rest = 1 - total
        if rest > 1e-9:
            freqs[remainder] = freqs.get(remainder, 0.0) + rest
        strategy[hand] = freqs
    return strategy


def build_node(
    *,
    key: str,
    stack_bb: float,
    sizings: dict[str, float],
    ranges: dict[str, str],
    remainder: str = "fold",
) -> dict[str, Any]:
    """Узел в каноническом формате; проверяется общей валидацией (`parse_chart_set`)."""
    return {
        "key": key,
        "stack_bb": stack_bb,
        "sizings": dict(sizings),
        "strategy": strategy_from_ranges(ranges, remainder=remainder),
    }


def _split_weight(token: str) -> tuple[str, float]:
    body, sep, raw = token.partition(":")
    if not sep:
        return body.strip(), 1.0
    try:
        weight = float(raw)
    except ValueError:
        raise RangeParseError(f"{token}: weight is not a number") from None
    if not 0 <= weight <= 1:
        raise RangeParseError(f"{token}: weight must be in [0, 1]")
    return body.strip(), weight


def _expand(body: str, token: str) -> list[frozenset[str]]:
    combo = _COMBO_RE.match(body)
    if combo:
        first, second = combo[1] + combo[2], combo[3] + combo[4]
        if first == second:
            raise RangeParseError(f"{token}: duplicate card")
        return [frozenset((first, second))]

    hands = _expand_classes(body, token)
    return [c for hand in hands for c in _class_combos(hand)]


def _expand_classes(body: str, token: str) -> list[str]:
    if body.endswith("+"):
        high, low, suffix = _parse_class(body[:-1], token)
        if high == low:
            return [r + r for r in _ASCENDING[_ASCENDING.index(low):]]
        kickers = _ASCENDING[_ASCENDING.index(low):_ASCENDING.index(high)]
        return [h for k in kickers for h in _with_suffix(high + k, suffix)]

    if "-" in body:
        left, _, right = body.partition("-")
        h1, l1, s1 = _parse_class(left, token)
        h2, l2, s2 = _parse_class(right, token)
        pairs = h1 == l1 and h2 == l2
        if not pairs and (h1 != h2 or s1 != s2 or h1 == l1 or h2 == l2):
            raise RangeParseError(f"{token}: range ends must share high card and suit")
        lo, hi = sorted((_ASCENDING.index(l1), _ASCENDING.index(l2)))
        if pairs:
            return [r + r for r in _ASCENDING[lo:hi + 1]]
        return [h for k in _ASCENDING[lo:hi + 1] for h in _with_suffix(h1 + k, s1)]

    high, low, suffix = _parse_class(body, token)
    if high == low:
        return [high + low]
    return _with_suffix(high + low, suffix)


def _parse_class(text: str, token: str) -> tuple[str, str, str]:
    match = _CLASS_RE.match(text)
    if not match:
        raise RangeParseError(f"{token}: unknown hand {text!r}")
    high, low, suffix = match.groups()
    if high == low and suffix:
        raise RangeParseError(f"{token}: pairs have no suit suffix")
    if RANKS.index(high) > RANKS.index(low):
        raise RangeParseError(f"{token}: higher card must come first")
    return high, low, suffix


def _with_suffix(ranks: str, suffix: str) -> list[str]:
    return [ranks + s for s in (suffix,) if s] or [ranks + "s", ranks + "o"]


def _class_combos(hand: str) -> list[frozenset[str]]:
    high, low = hand[0], hand[1]
    if high == low:
        return [frozenset((high + a, low + b)) for a, b in combinations(SUITS, 2)]
    suited = hand.endswith("s")
    return [
        frozenset((high + a, low + b))
        for a, b in product(SUITS, SUITS)
        if (a == b) == suited
    ]


def _class_of(combo: frozenset[str]) -> str:
    high, low = sorted(combo, key=lambda card: RANKS.index(card[0]))
    if high[0] == low[0]:
        return high[0] + low[0]
    return high[0] + low[0] + ("s" if high[1] == low[1] else "o")
