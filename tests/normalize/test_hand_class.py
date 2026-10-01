import pytest

from core.normalize.hand_class import ALL_HAND_CLASSES, hand_class


@pytest.mark.parametrize(
    "cards, expected",
    [
        ("AhKh", "AKs"),
        ("7c7d", "77"),
        ("Td9s", "T9o"),
        ("KhAh", "AKs"),  # порядок карт не важен
        ("9sTd", "T9o"),
        ("2c3c", "32s"),
        ("Ah Kd", "AKo"),  # пробел между картами допустим
    ],
)
def test_hand_class(cards, expected):
    assert hand_class(cards) == expected


@pytest.mark.parametrize("cards", ["AhAh", "1hKh", "AxKh", "AhK", "AhKhQh", ""])
def test_invalid_cards_are_rejected(cards):
    with pytest.raises(ValueError):
        hand_class(cards)


def test_there_are_169_hand_classes():
    assert len(ALL_HAND_CLASSES) == 169
    assert len(set(ALL_HAND_CLASSES)) == 169
    assert {"AA", "AKs", "AKo", "32o", "22"} <= set(ALL_HAND_CLASSES)
