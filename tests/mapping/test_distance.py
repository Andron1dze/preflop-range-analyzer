import math

import pytest

from core.mapping.distance import MappingConfig, NodeCandidate, match_decision, size_distance, stack_distance, weight

SIGMA = math.log(1.25)
CONFIG = MappingConfig()

BTN_RFI = NodeCandidate(node_id=1, stack_bb=40, sizings={"BTN": 2.0})
BB_VS_BTN = NodeCandidate(node_id=2, stack_bb=40, sizings={"BTN": 2.0, "BB": 8.0})


def test_default_config():
    assert CONFIG.sigma == pytest.approx(SIGMA)
    assert CONFIG.cutoff == 0.1


def test_version_encodes_parameters():
    assert MappingConfig().version(chart_set_id=3) == "grid-v1/cs3/sigma=0.223144/cutoff=0.1"
    assert MappingConfig(cutoff=0.2).version(chart_set_id=3) != MappingConfig().version(chart_set_id=3)


# --- компоненты расстояния ---------------------------------------------------


@pytest.mark.parametrize("eff, expected", [(40, 0.0), (50, math.log(1.25) / SIGMA), (32, math.log(0.8) / SIGMA)])
def test_stack_distance(eff, expected):
    assert stack_distance(eff, 40, CONFIG) == pytest.approx(expected)


def test_stack_distance_is_symmetric_in_log_space():
    assert abs(stack_distance(32, 40, CONFIG)) == pytest.approx(abs(stack_distance(50, 40, CONFIG)))


def test_size_distance():
    assert size_distance(2.5, 2.0, CONFIG) == pytest.approx(1.0)
    assert size_distance(2.0, 2.0, CONFIG) == 0.0


# --- вес: примеры из Q4 ------------------------------------------------------


@pytest.mark.parametrize(
    "eff, open_size, d, w",
    [
        (40, 2.0, 0.00, 1.00),
        (40, 2.2, 0.43, 0.91),
        (40, 2.5, 1.00, 0.61),
        (40, 3.0, 1.82, 0.19),
        (35, 2.0, 0.60, 0.84),
        (30, 2.0, 1.29, 0.44),
        (25, 2.0, 2.11, 0.11),
        (50, 2.0, 1.00, 0.61),
        (60, 2.0, 1.82, 0.19),
        (30, 2.5, 1.63, 0.26),
    ],
)
def test_weight_examples_from_q4(eff, open_size, d, w):
    match = match_decision(f"UTG:F,BTN:R{open_size:g}", eff, [BB_VS_BTN], CONFIG)
    assert match.node_id == 2
    assert match.distance == pytest.approx(d, abs=0.005)
    assert match.weight == pytest.approx(w, abs=0.005)


def test_far_decision_gets_zero_weight_but_keeps_distance():
    # Открытие 3.5bb: w = 0.04 < 0.1.
    match = match_decision("BTN:R3.5", 40, [BB_VS_BTN], CONFIG)
    assert match.distance == pytest.approx(2.51, abs=0.005)
    assert match.weight == 0.0


def test_weight_function():
    assert weight(0.0, CONFIG) == 1.0
    assert weight(1.0, CONFIG) == pytest.approx(math.exp(-0.5))
    assert weight(3.0, CONFIG) == 0.0


# --- что входит в расстояние -------------------------------------------------


def test_current_hero_raise_is_not_part_of_distance():
    # Линия — только действия ДО решения; собственный размер открытия героя не влияет.
    match = match_decision("UTG:F,UTG1:F,LJ:F,HJ:F,CO:F", 40, [BTN_RFI], CONFIG)
    assert (match.distance, match.weight) == (0.0, 1.0)


def test_previous_raises_including_heros_are_part_of_distance():
    node = NodeCandidate(node_id=3, stack_bb=40, sizings={"CO": 2.0, "BTN": 5.4})
    # Герой на CO открылся на 2.5 (а не 2.0), BTN 3-бетит на 5.4 — герой отвечает на 3-бет.
    match = match_decision("CO:R2.5,BTN:R5.4", 40, [node], CONFIG)
    assert match.distance == pytest.approx(1.0)


def test_jam_against_jam_has_zero_size_distance():
    node = NodeCandidate(node_id=4, stack_bb=40, sizings={"CO": 2.0})
    match = match_decision("CO:R2,BTN:R40ai", 40, [node], CONFIG)
    assert match.distance == 0.0


def test_last_raise_of_position_is_compared():
    # Формат чарта хранит один сайзинг на позицию — сравниваем последний рейз позиции.
    node = NodeCandidate(node_id=5, stack_bb=40, sizings={"CO": 12.0, "BTN": 5.4})
    match = match_decision("CO:R2,BTN:R5.4,CO:R12", 40, [node], CONFIG)
    assert match.distance == 0.0


def test_node_without_sizing_for_a_raiser_does_not_match():
    node = NodeCandidate(node_id=6, stack_bb=40, sizings={"BB": 8.0})
    assert match_decision("BTN:R2", 40, [node], CONFIG) is None


def test_nearest_stack_depth_wins():
    deep = NodeCandidate(node_id=7, stack_bb=60, sizings={"BTN": 2.0})
    shallow = NodeCandidate(node_id=8, stack_bb=40, sizings={"BTN": 2.0})
    assert match_decision("", 45, [deep, shallow], CONFIG).node_id == 8
    assert match_decision("", 55, [deep, shallow], CONFIG).node_id == 7


def test_no_candidates():
    assert match_decision("", 40, [], CONFIG) is None
