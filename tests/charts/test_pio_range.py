import pytest

from core.charts.format import SYNTHETIC_SOURCE, parse_chart_set
from core.charts.importers.pio_range import RangeParseError, build_node, parse_range, strategy_from_ranges

# --- одна строка диапазона ---------------------------------------------------


def test_class_weights_and_missing_hands_are_zero():
    weights = parse_range("AA,AKs:0.5,KQo:0.25")
    assert len(weights) == 169
    assert weights["AA"] == 1.0
    assert weights["AKs"] == 0.5
    assert weights["KQo"] == 0.25
    assert weights["AKo"] == 0.0
    assert weights["72o"] == 0.0


def test_empty_string_and_whitespace():
    assert set(parse_range("").values()) == {0.0}
    assert parse_range("  AA , KK:0.5 \n")["KK"] == 0.5


@pytest.mark.parametrize(
    "text, hand, expected",
    [
        ("AhKh:0.5", "AKs", 0.5 / 4),  # одно из 4 комбо, сыграно наполовину
        ("AhKh,AdKd,AcKc,AsKs", "AKs", 1.0),
        ("AhKh:0.5,AsKs", "AKs", 1.5 / 4),
        ("KhAh", "AKs", 1 / 4),  # порядок карт не важен
        ("AhKd", "AKo", 1 / 12),
        ("AhAd", "AA", 1 / 6),
        ("AhKh,AKo:0.5", "AKo", 0.5),
    ],
)
def test_combos_are_averaged_over_hand_class(text, hand, expected):
    assert parse_range(text)[hand] == pytest.approx(expected)


def test_combo_does_not_leak_into_other_class():
    weights = parse_range("AhKh")
    assert weights["AKo"] == 0.0


@pytest.mark.parametrize(
    "text, included, excluded",
    [
        ("AK", {"AKs", "AKo"}, {"AQs"}),
        ("22+", {"22", "77", "AA"}, {"AKs"}),
        ("99-66", {"99", "88", "77", "66"}, {"TT", "55"}),
        ("A2s+", {"A2s", "A9s", "AKs"}, {"AKo", "KQs"}),
        ("ATo+", {"ATo", "AJo", "AKo"}, {"A9o", "ATs"}),
        ("KTo-K8o", {"KTo", "K9o", "K8o"}, {"KJo", "K7o", "KTs"}),
        ("T8s+", {"T8s", "T9s"}, {"T7s", "JTs"}),
    ],
)
def test_range_shorthands(text, included, excluded):
    weights = parse_range(text + ":0.5")
    assert {h for h in included if weights[h] == 0.5} == included
    assert {h for h in excluded if weights[h] == 0.0} == excluded


@pytest.mark.parametrize(
    "text, message",
    [
        ("AKx", "AKx"),
        ("AA:1.5", "AA:1.5"),
        ("AA:-0.1", "AA:-0.1"),
        ("AA:half", "AA:half"),
        ("AhAh", "AhAh"),
        ("KTo-Q8o", "KTo-Q8o"),
        ("AA,AA:0.5", "AA"),  # одно комбо указано дважды
        ("AKs,AhKh", "AhKh"),
    ],
)
def test_invalid_tokens_are_rejected(text, message):
    with pytest.raises(RangeParseError, match=message):
        parse_range(text)


# --- несколько действий → стратегия узла -------------------------------------


def test_remainder_goes_to_fold():
    strategy = strategy_from_ranges({"raise": "AA,AKs:0.5,KQo:0.25"})
    assert strategy["AA"] == {"raise": 1.0}
    assert strategy["AKs"] == pytest.approx({"raise": 0.5, "fold": 0.5})
    assert strategy["KQo"] == pytest.approx({"raise": 0.25, "fold": 0.75})
    assert strategy["72o"] == {"fold": 1.0}


def test_several_actions_with_remainder():
    strategy = strategy_from_ranges({"raise": "AA,AKs:0.5", "call": "AKs:0.25,QQ"})
    assert strategy["AKs"] == pytest.approx({"raise": 0.5, "call": 0.25, "fold": 0.25})
    assert strategy["QQ"] == {"call": 1.0}


def test_remainder_action_can_be_changed():
    # Например, BB против лимпа: остаток — чек, а не фолд.
    strategy = strategy_from_ranges({"raise": "AA"}, remainder="check")
    assert strategy["72o"] == {"check": 1.0}


def test_actions_exceeding_one_are_rejected():
    with pytest.raises(RangeParseError, match="AKs"):
        strategy_from_ranges({"raise": "AKs:0.75", "call": "AKs:0.5"})


def test_error_names_the_action():
    with pytest.raises(RangeParseError, match="call"):
        strategy_from_ranges({"raise": "AA", "call": "AKx"})


# --- результат проходит валидацию канонического формата ----------------------


def test_built_node_passes_chart_validation():
    node = build_node(
        key="40bb|HERO=BTN",
        stack_bb=40,
        sizings={"BTN": 2.2},
        ranges={"raise": "22+,A2s+,ATo+,KTs+,KhQd:0.5", "allin": "QhJd:0.5"},
    )
    chart = parse_chart_set(
        {
            "chart_set": {
                "model": "chipev",
                "ante": {"type": "bb", "size_bb": 1.0},
                "source": SYNTHETIC_SOURCE,
                "version": "1",
            },
            "nodes": [node],
        }
    )
    strategy = chart.nodes[0].strategy
    assert strategy["AA"] == {"raise": 1.0}
    assert strategy["72o"] == {"fold": 1.0}
    assert strategy["KQo"] == pytest.approx({"raise": 0.5 / 12, "fold": 11.5 / 12})
    assert strategy["QJo"] == pytest.approx({"allin": 0.5 / 12, "fold": 11.5 / 12})
