import pytest

from core.model.positions import assign_positions


def test_full_8max_table():
    positions = assign_positions(seats=[1, 2, 3, 4, 5, 6, 7, 8], button_seat=6)
    assert positions == {
        7: "SB", 8: "BB", 1: "UTG", 2: "UTG1", 3: "LJ", 4: "HJ", 5: "CO", 6: "BTN",
    }


def test_short_handed_table_drops_early_positions():
    positions = assign_positions(seats=[1, 3, 4, 6, 7, 8], button_seat=1)
    assert positions == {3: "SB", 4: "BB", 6: "LJ", 7: "HJ", 8: "CO", 1: "BTN"}


def test_three_handed():
    assert assign_positions(seats=[2, 5, 8], button_seat=5) == {8: "SB", 2: "BB", 5: "BTN"}


def test_heads_up_button_is_small_blind():
    assert assign_positions(seats=[2, 5], button_seat=5) == {5: "SB", 2: "BB"}


def test_button_on_empty_seat_is_rejected():
    with pytest.raises(ValueError):
        assign_positions(seats=[1, 2, 3], button_seat=4)


def test_more_than_8_players_is_rejected():
    with pytest.raises(ValueError):
        assign_positions(seats=list(range(1, 10)), button_seat=1)
