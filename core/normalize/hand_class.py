RANKS = "AKQJT98765432"
SUITS = "cdhs"


def _all_hand_classes() -> tuple[str, ...]:
    classes = []
    for i, high in enumerate(RANKS):
        for j, low in enumerate(RANKS):
            if i == j:
                classes.append(high + low)
            elif i < j:
                classes.append(f"{high}{low}s")
                classes.append(f"{high}{low}o")
    return tuple(classes)


ALL_HAND_CLASSES = _all_hand_classes()


def hand_class(cards: str) -> str:
    """Две карты ("AhKh", "Kh Ah") → класс руки из 169 ("AKs")."""
    compact = cards.replace(" ", "")
    if len(compact) != 4:
        raise ValueError(f"expected two cards, got {cards!r}")
    first, second = compact[:2], compact[2:]
    for card in (first, second):
        if card[0] not in RANKS or card[1] not in SUITS:
            raise ValueError(f"invalid card {card!r} in {cards!r}")
    if first == second:
        raise ValueError(f"duplicate card in {cards!r}")

    high, low = sorted((first, second), key=lambda c: RANKS.index(c[0]))
    if high[0] == low[0]:
        return high[0] + low[0]
    return high[0] + low[0] + ("s" if high[1] == low[1] else "o")
