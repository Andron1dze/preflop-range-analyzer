POSITIONS_8MAX = ("UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB")

# Позиции до блайндов от самой ранней; при неполном столе ранние выпадают первыми.
_EARLY_TO_BUTTON = POSITIONS_8MAX[:6]


def assign_positions(seats: list[int], button_seat: int) -> dict[int, str]:
    """Номер места → позиция. Хедз-ап: баттон ставит SB."""
    if len(seats) > len(POSITIONS_8MAX):
        raise ValueError(f"too many players for 8-max: {len(seats)}")
    if button_seat not in seats:
        raise ValueError(f"button seat {button_seat} is empty")

    ordered = sorted(seats)
    start = ordered.index(button_seat) + 1
    # По часовой стрелке от места после баттона; баттон — последним.
    clockwise = ordered[start:] + ordered[:start]

    if len(seats) == 2:
        return {clockwise[1]: "SB", clockwise[0]: "BB"}

    names = ["SB", "BB", *_EARLY_TO_BUTTON[len(_EARLY_TO_BUTTON) - (len(seats) - 2):]]
    return dict(zip(clockwise, names))
