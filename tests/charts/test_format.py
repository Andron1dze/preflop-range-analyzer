"""Канонический формат чартов. Частоты в тестах выдуманы и отношения к GTO не имеют."""

import copy
import json

import pytest
import yaml

from core.charts.format import SYNTHETIC_SOURCE, ChartValidationError, load_chart_file, parse_chart_set
from core.normalize.hand_class import ALL_HAND_CLASSES


def chart_data(source=SYNTHETIC_SOURCE):
    strategy = {hand: {"fold": 0.5, "raise": 0.5} for hand in ALL_HAND_CLASSES}
    strategy["AA"] = {"raise": 1.0}
    return {
        "chart_set": {
            "model": "chipev",
            "ante": {"type": "bb", "size_bb": 1.0},
            "source": source,
            "version": "1",
        },
        "nodes": [
            {
                "key": "40bb|HERO=BTN",
                "stack_bb": 40,
                "sizings": {"BTN": 2.2},
                "strategy": strategy,
            }
        ],
    }


def errors_of(data):
    with pytest.raises(ChartValidationError) as info:
        parse_chart_set(data)
    return "\n".join(info.value.errors)


# --- успешная загрузка -------------------------------------------------------


def test_valid_chart_set_is_parsed():
    chart = parse_chart_set(chart_data())
    assert chart.model == "chipev"
    assert (chart.ante_type, chart.ante_size_bb) == ("bb", 1.0)
    [node] = chart.nodes
    assert node.key == "40bb|HERO=BTN"
    assert node.hero_position == "BTN"
    assert node.stack_bb == 40
    assert node.sizings == {"BTN": 2.2}
    assert len(node.strategy) == 169
    # Отсутствующее действие — частота 0.
    assert node.strategy["AA"] == {"raise": 1.0}


@pytest.mark.parametrize("suffix, dump", [(".json", json.dumps), (".yaml", yaml.safe_dump), (".yml", yaml.safe_dump)])
def test_chart_file_is_loaded_by_extension(tmp_path, suffix, dump):
    path = tmp_path / f"chart{suffix}"
    path.write_text(dump(chart_data()), encoding="utf-8")
    assert load_chart_file(path).nodes[0].key == "40bb|HERO=BTN"


def test_unknown_file_extension_is_rejected(tmp_path):
    path = tmp_path / "chart.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ChartValidationError, match="extension"):
        load_chart_file(path)


def test_frequencies_within_tolerance_are_renormalized():
    data = chart_data()
    data["nodes"][0]["strategy"]["KK"] = {"fold": 0.33, "call": 0.33, "raise": 0.335}
    kk = parse_chart_set(data, tolerance=0.01).nodes[0].strategy["KK"]
    assert sum(kk.values()) == pytest.approx(1.0, abs=1e-12)
    assert kk["raise"] == pytest.approx(0.335 / 0.995)


# --- сумма частот ------------------------------------------------------------


@pytest.mark.parametrize("freqs", [{"fold": 0.5, "raise": 0.4}, {"fold": 0.6, "raise": 0.6}])
def test_frequencies_must_sum_to_one(freqs):
    data = chart_data()
    data["nodes"][0]["strategy"]["72o"] = freqs
    message = errors_of(data)
    assert "72o" in message and "sum" in message


def test_tolerance_is_configurable():
    data = chart_data()
    data["nodes"][0]["strategy"]["72o"] = {"fold": 0.5, "raise": 0.49}
    parse_chart_set(data, tolerance=0.02)
    with pytest.raises(ChartValidationError):
        parse_chart_set(data, tolerance=0.001)


@pytest.mark.parametrize("freq", [-0.1, 1.5, "half"])
def test_frequency_must_be_a_number_in_unit_interval(freq):
    data = chart_data()
    data["nodes"][0]["strategy"]["72o"] = {"fold": freq, "raise": 0.5}
    assert "72o" in errors_of(data)


def test_unknown_action_is_rejected():
    data = chart_data()
    data["nodes"][0]["strategy"]["72o"] = {"fold": 0.5, "limp": 0.5}
    assert "limp" in errors_of(data)


# --- покрытие 169 классов ----------------------------------------------------


def test_all_169_hand_classes_must_be_covered():
    data = chart_data()
    del data["nodes"][0]["strategy"]["T9s"]
    del data["nodes"][0]["strategy"]["22"]
    message = errors_of(data)
    assert "T9s" in message and "22" in message


def test_unknown_hand_class_is_rejected():
    data = chart_data()
    data["nodes"][0]["strategy"]["AKx"] = {"fold": 1.0}
    assert "AKx" in errors_of(data)


# --- сайзинги и анте ---------------------------------------------------------


@pytest.mark.parametrize("sizings", [None, {}])
def test_sizings_are_required(sizings):
    data = chart_data()
    if sizings is None:
        del data["nodes"][0]["sizings"]
    else:
        data["nodes"][0]["sizings"] = sizings
    assert "sizings" in errors_of(data)


def test_raise_in_strategy_requires_hero_sizing():
    data = chart_data()
    data["nodes"][0]["key"] = "40bb|CO:open|HERO=BB"
    data["nodes"][0]["sizings"] = {"CO": 2.2}
    message = errors_of(data)
    assert "BB" in message and "sizing" in message


@pytest.mark.parametrize("size", [0, -2.2, "big"])
def test_sizing_must_be_positive_number(size):
    data = chart_data()
    data["nodes"][0]["sizings"] = {"BTN": size}
    assert "sizing" in errors_of(data)


def test_ante_structure_is_required():
    data = chart_data()
    del data["chart_set"]["ante"]
    assert "ante" in errors_of(data)


@pytest.mark.parametrize(
    "ante",
    [
        {"type": "bb"},  # нет размера
        {"size_bb": 1.0},  # нет типа
        {"type": "straddle", "size_bb": 1.0},
        {"type": "bb", "size_bb": -1},
        {"type": "none", "size_bb": 1.0},  # нет анте — размер должен быть 0
        "bb_ante",
    ],
)
def test_ante_structure_must_be_complete(ante):
    data = chart_data()
    data["chart_set"]["ante"] = ante
    assert "ante" in errors_of(data)


# --- прочая структура --------------------------------------------------------


@pytest.mark.parametrize("field", ["model", "source", "version"])
def test_chart_set_fields_are_required(field):
    data = chart_data()
    del data["chart_set"][field]
    assert field in errors_of(data)


@pytest.mark.parametrize("model", ["icm", "ICM:ft", "cev"])
def test_chart_set_model_must_be_known(model):
    data = chart_data()
    data["chart_set"]["model"] = model
    assert "model" in errors_of(data)


@pytest.mark.parametrize("key", ["40bb|CO:open", "40bb|HERO=XX", ""])
def test_node_key_must_name_hero_position(key):
    data = chart_data()
    data["nodes"][0]["key"] = key
    assert "key" in errors_of(data)


@pytest.mark.parametrize("stack", [0, -40, None, "deep"])
def test_stack_must_be_positive(stack):
    data = chart_data()
    data["nodes"][0]["stack_bb"] = stack
    assert "stack_bb" in errors_of(data)


def test_duplicate_node_is_rejected():
    data = chart_data()
    data["nodes"].append(copy.deepcopy(data["nodes"][0]))
    assert "duplicate" in errors_of(data)


def test_all_errors_are_reported_at_once():
    data = chart_data()
    del data["chart_set"]["ante"]
    del data["nodes"][0]["strategy"]["T9s"]
    data["nodes"][0]["strategy"]["72o"] = {"fold": 0.1}
    message = errors_of(data)
    assert "ante" in message and "T9s" in message and "72o" in message


# --- синтетические чарты -----------------------------------------------------


def test_synthetic_chart_loads_but_is_not_eligible_for_analysis():
    chart = parse_chart_set(chart_data(source=SYNTHETIC_SOURCE))
    assert chart.eligible_for_analysis is False


def test_real_source_is_eligible_for_analysis():
    chart = parse_chart_set(chart_data(source="gto-wizard-export-2026-09"))
    assert chart.eligible_for_analysis is True
