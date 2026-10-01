"""Префлоп-раздача → точки решения в нормализованных числах.

Соглашения (маппинг на узлы чартов должен им соответствовать):
- size_bb: итоговая ставка игрока после действия (raise *to* / сумма колла), в bb.
- size_pot: только для рейза; как у солверов — прибавка сверх колла, делённая
  на пот после колла. 1.0 = пот-сайз рейз.
- eff_stack_bb: min(свой стек, наибольший стек среди не сбросивших соперников);
  стеки — на начало раздачи минус уплаченное анте (блайнды остаются в стеке).
"""

from dataclasses import dataclass

from core.model import ActionKind, AnteType, Hand, assign_positions

_LINE_CODES = {
    ActionKind.FOLD: "F",
    ActionKind.CHECK: "X",
    ActionKind.CALL: "C",
    ActionKind.RAISE: "R",
}


@dataclass(frozen=True)
class DecisionPoint:
    player: str
    position: str
    kind: ActionKind
    size_bb: float | None
    size_pot: float | None
    eff_stack_bb: float
    # Пот перед решением, включая анте и блайнды.
    pot_bb: float
    all_in: bool
    # Предыдущие действия раздачи, например "UTG:F,CO:R2.2,BTN:R7".
    line: str


def ante_paid(hand: Hand) -> dict[str, float]:
    positions = _positions_by_player(hand)
    paid = {}
    for seat in hand.seats:
        if hand.ante_type is AnteType.EACH or (
            hand.ante_type is AnteType.BB and positions[seat.player] == "BB"
        ):
            paid[seat.player] = min(hand.ante, seat.stack)
        else:
            paid[seat.player] = 0.0
    return paid


def normalize(hand: Hand) -> list[DecisionPoint]:
    bb = hand.big_blind
    positions = _positions_by_player(hand)
    antes = ante_paid(hand)
    # Сколько фишек игрок может поставить на префлопе (уже без анте).
    stacks = {s.player: s.stack - antes[s.player] for s in hand.seats}

    committed = {player: 0.0 for player in stacks}
    for player, position in positions.items():
        if position == "SB":
            committed[player] = min(hand.small_blind, stacks[player])
        elif position == "BB":
            committed[player] = min(bb, stacks[player])
    pot = sum(antes.values()) + sum(committed.values())
    current_bet = max(committed.values())

    folded: set[str] = set()
    tokens: list[str] = []
    points = []
    for index, action in enumerate(hand.actions, start=1):
        player = action.player
        where = f"action #{index} by {player}"
        if player not in stacks:
            raise ValueError(f"{where}: player is not seated")
        if player in folded:
            raise ValueError(f"{where}: player has already folded")
        opponents = [p for p in stacks if p != player and p not in folded]
        if not opponents:
            raise ValueError(f"{where}: hand is already over")

        eff_stack = min(stacks[player], max(stacks[p] for p in opponents))
        size_pot = None
        if action.kind is ActionKind.FOLD:
            folded.add(player)
            to = None
        elif action.kind is ActionKind.CHECK:
            if committed[player] < current_bet:
                raise ValueError(f"{where}: cannot check facing a bet")
            to = None
        elif action.kind is ActionKind.CALL:
            to = min(current_bet, stacks[player])
        else:
            to = action.to
            if to is None or to <= current_bet:
                raise ValueError(f"{where}: raise to {to} does not exceed bet {current_bet}")
            if to > stacks[player]:
                raise ValueError(f"{where}: raise to {to} exceeds stack {stacks[player]}")
            call = current_bet - committed[player]
            size_pot = (to - current_bet) / (pot + call)

        all_in = to is not None and to == stacks[player]
        size_bb = None if to is None else to / bb
        points.append(
            DecisionPoint(
                player=player,
                position=positions[player],
                kind=action.kind,
                size_bb=size_bb,
                size_pot=size_pot,
                eff_stack_bb=eff_stack / bb,
                pot_bb=pot / bb,
                all_in=all_in,
                line=",".join(tokens),
            )
        )

        token = f"{positions[player]}:{_LINE_CODES[action.kind]}"
        if action.kind is ActionKind.RAISE:
            token += f"{size_bb:g}"
        if all_in:
            token += "ai"
        tokens.append(token)
        if to is not None:
            pot += to - committed[player]
            committed[player] = to
            current_bet = max(current_bet, to)
    return points


def _positions_by_player(hand: Hand) -> dict[str, str]:
    by_seat = assign_positions([s.number for s in hand.seats], hand.button_seat)
    return {s.player: by_seat[s.number] for s in hand.seats}
