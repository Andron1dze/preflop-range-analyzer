"""Внутренняя модель раздачи (по образцу PHH). Суммы — в фишках, как в источнике."""

from dataclasses import dataclass, field
from enum import StrEnum

STAGES = ("early", "mid", "bubble", "itm", "ft")
STAGE_SOURCES = ("proxy", "summary", "manual")


class AnteType(StrEnum):
    NONE = "none"
    EACH = "each"  # анте платит каждый игрок
    BB = "bb"  # BB-ante: большой блайнд платит анте за весь стол


class ActionKind(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"  # включает открытие и пуш; размер — в Action.to


@dataclass
class Seat:
    number: int
    player: str
    # Стек на начало раздачи, до анте и блайндов.
    stack: float


@dataclass
class Action:
    player: str
    kind: ActionKind
    # Для рейза: итоговая ставка игрока на префлопе (raise *to*), без анте.
    # Для колла не нужна: сумма выводится из текущей ставки и стека.
    to: float | None = None


@dataclass
class Hand:
    external_id: str | None
    tournament_id: str | None
    level: int | None
    small_blind: float
    big_blind: float
    ante: float
    ante_type: AnteType
    button_seat: int
    seats: list[Seat]
    hero: str
    hero_cards: str | None
    # Только префлоп, по порядку; блайнды и анте сюда не входят.
    actions: list[Action] = field(default_factory=list)
    stage: str | None = None
    stage_source: str | None = None


@dataclass
class ParsedHand:
    """Результат любого адаптера: раздача и её сырьё для ре-парсинга."""

    hand: Hand
    # Текст HH или сериализованная форма ручного ввода.
    raw_text: str
