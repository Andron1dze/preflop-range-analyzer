"""Линия решения (`decisions.line`) → абстрактная линия для ключа узла.

Ключ узла — позиции и тип действия без размеров (тикет 15): первый рейз — open,
второй — 3bet, третий — 4bet, дальше — 5bet; олл-ин-рейз — всегда jam; колл
без рейза — limp. Сбросы и чеки в ключ не входят.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LineToken:
    position: str
    code: str  # F, X, C, R
    size_bb: float | None  # только для рейза
    all_in: bool


_RAISE_TYPES = ("open", "3bet", "4bet")


def parse_line(line: str) -> list[LineToken]:
    tokens = []
    for raw in filter(None, line.split(",")):
        position, action = raw.split(":", 1)
        code, rest = action[0], action[1:]
        all_in = rest.endswith("ai")
        rest = rest.removesuffix("ai")
        size = float(rest) if code == "R" and rest else None
        tokens.append(LineToken(position, code, size, all_in))
    return tokens


def _raise_type(number: int, all_in: bool) -> str:
    if all_in:
        return "jam"
    return _RAISE_TYPES[number - 1] if number <= len(_RAISE_TYPES) else f"{number + 1}bet"


def abstract_line(line: str) -> str:
    parts = []
    raises = 0
    for token in parse_line(line):
        if token.code == "R":
            raises += 1
            parts.append(f"{token.position}:{_raise_type(raises, token.all_in)}")
        elif token.code == "C":
            parts.append(f"{token.position}:{'call' if raises else 'limp'}")
    return ",".join(parts)


def facing_action(line: str) -> str:
    """Тип последнего рейза перед решением: none / open / 3bet / 4bet / jam (5bet+ → 4bet)."""
    raises = [t for t in parse_line(line) if t.code == "R"]
    if not raises:
        return "none"
    return _raise_type(min(len(raises), len(_RAISE_TYPES)), raises[-1].all_in)


def node_key(stack_label: str, actions: str, hero_position: str) -> str:
    return "|".join(part for part in (stack_label, actions, f"HERO={hero_position}") if part)


def split_node_key(key: str) -> tuple[str, str, str]:
    parts = key.split("|")
    return parts[0], "|".join(parts[1:-1]), parts[-1].removeprefix("HERO=")
