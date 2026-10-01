import pytest

from core.mapping.line import abstract_line, facing_action, node_key, parse_line, split_node_key


def test_parse_line_tokens():
    tokens = parse_line("UTG:F,CO:R2.2,BTN:C2.2,BB:R40ai")
    assert [(t.position, t.code, t.size_bb, t.all_in) for t in tokens] == [
        ("UTG", "F", None, False),
        ("CO", "R", 2.2, False),
        ("BTN", "C", None, False),
        ("BB", "R", 40.0, True),
    ]
    assert parse_line("") == []


@pytest.mark.parametrize(
    "line, actions",
    [
        ("", ""),
        ("UTG:F,UTG1:F,LJ:F", ""),  # сбросы в ключ не входят
        ("UTG:F,CO:R2.2", "CO:open"),
        ("CO:R2,BTN:C2", "CO:open,BTN:call"),
        ("CO:R2,BTN:R5.4,SB:F,BB:F", "CO:open,BTN:3bet"),
        ("CO:R2,BTN:R5.4,CO:R12", "CO:open,BTN:3bet,CO:4bet"),
        ("CO:R2,BTN:R5.4,CO:R40ai", "CO:open,BTN:3bet,CO:jam"),  # олл-ин — всегда jam
        ("UTG:R40ai", "UTG:jam"),
        ("UTG:C1", "UTG:limp"),  # колл без рейза — лимп
        ("SB:C1,BB:X", "SB:limp"),
    ],
)
def test_abstract_line(line, actions):
    assert abstract_line(line) == actions


def test_node_key_round_trip():
    assert node_key("40bb", "", "BTN") == "40bb|HERO=BTN"
    assert node_key("40bb", "BTN:open", "BB") == "40bb|BTN:open|HERO=BB"
    assert split_node_key("40bb|BTN:open|HERO=BB") == ("40bb", "BTN:open", "BB")
    assert split_node_key("40bb|HERO=BTN") == ("40bb", "", "BTN")


@pytest.mark.parametrize(
    "line, facing",
    [
        ("", "none"),
        ("UTG:F,UTG1:F", "none"),
        ("UTG:C1", "none"),  # против лимпа рейза нет
        ("CO:R2.2", "open"),
        ("CO:R2.2,BTN:C2.2", "open"),
        ("CO:R2,BTN:R5.4", "3bet"),
        ("CO:R2,BTN:R5.4,CO:R12", "4bet"),
        ("CO:R2,BTN:R40ai", "jam"),
    ],
)
def test_facing_action(line, facing):
    assert facing_action(line) == facing
