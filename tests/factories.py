
from core.model import Action, ActionKind, AnteType, Hand, Seat

# Места 1..8, баттон на 6: UTG=p1, UTG1=p2, LJ=p3, HJ=p4, CO=p5, BTN=p6, SB=p7, BB=p8.


def make_hand(actions, *, stacks=None, ante=100.0, ante_type=AnteType.BB, hero="p6"):
    stacks = stacks or {}
    return Hand(
        external_id="1",
        tournament_id="T1",
        level=5,
        small_blind=50.0,
        big_blind=100.0,
        ante=ante,
        ante_type=ante_type,
        button_seat=6,
        seats=[Seat(n, f"p{n}", stacks.get(f"p{n}", 4000.0)) for n in range(1, 9)],
        hero=hero,
        hero_cards="AhKh",
        actions=[Action(player, ActionKind(kind), to) for player, kind, to in actions],
    )


def folds(*players):
    return [(p, "fold", None) for p in players]


