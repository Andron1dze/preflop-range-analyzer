import numpy as np
import pytest

from core.analysis.config import AnalysisConfig
from core.analysis.features import (
    FEATURE_LEVELS,
    encode,
    feature_values,
    stack_bucket,
    translate_action,
)

CONFIG = AnalysisConfig()

# --- бакеты ------------------------------------------------------------------


@pytest.mark.parametrize(
    "eff, bucket",
    [(5, "<30"), (29.99, "<30"), (30, "30-35"), (34.9, "30-35"), (35, "35-45"), (40, "35-45"), (45, "45+"), (120, "45+")],
)
def test_stack_bucket(eff, bucket):
    assert stack_bucket(eff, CONFIG.stack_edges) == bucket


def test_bucket_labels_follow_config():
    config = AnalysisConfig(stack_edges=(15, 25))
    assert [stack_bucket(x, config.stack_edges) for x in (10, 20, 30)] == ["<15", "15-25", "25+"]


# --- перевод действия в словарь чарта ----------------------------------------

THREE_WAY = frozenset({"fold", "call", "raise"})
WITH_JAM = frozenset({"fold", "call", "raise", "allin"})
JAM_ONLY = frozenset({"fold", "allin"})


@pytest.mark.parametrize(
    "action, all_in, node_actions, expected",
    [
        ("fold", False, THREE_WAY, "fold"),
        ("call", False, THREE_WAY, "call"),
        ("call", True, THREE_WAY, "call"),  # колл на все — всё равно колл
        ("raise", False, WITH_JAM, "raise"),
        ("raise", True, WITH_JAM, "allin"),
        ("raise", True, THREE_WAY, "raise"),  # в узле нет пуша — это рейз
        ("raise", False, JAM_ONLY, "allin"),  # в узле только пуш — рейз считается пушем
        ("check", False, THREE_WAY, None),  # действия нет в узле — решение не сопоставлено
    ],
)
def test_translate_action(action, all_in, node_actions, expected):
    assert translate_action(action, all_in, node_actions) == expected


# --- значения признаков ------------------------------------------------------


def test_feature_values():
    values = feature_values(
        position="BB", eff_stack_bb=24.0, line="UTG:F,CO:R2.2", hand_class="98s", p_step=0.95,
        stage=None, config=CONFIG,
    )
    assert values == {
        "position": "BB",
        "stack": "<30",
        "facing": "open",
        "hand_group": "suited_connector",
        "zone": "pure",
        "stage": "unknown",
    }


# --- матрица признаков -------------------------------------------------------


def row(**overrides):
    base = {"position": "UTG", "stack": "35-45", "facing": "none", "hand_group": "pair", "zone": "mix", "stage": "unknown"}
    base.update(overrides)
    return base


def test_reference_levels_come_first():
    assert FEATURE_LEVELS["position"][0] == "UTG"
    assert FEATURE_LEVELS["stack"] == ["35-45", "<30", "30-35", "45+"]  # номинальный 40bb — первым
    assert FEATURE_LEVELS["facing"] == ["none", "open", "3bet", "4bet", "jam"]
    assert FEATURE_LEVELS["zone"][0] == "mix"
    assert FEATURE_LEVELS["stage"][0] == "unknown"


def test_encode_one_hot_with_intercept_and_dropped_reference():
    X, names = encode([row(), row(position="BTN", stage="bubble"), row(stack="<30")])
    assert names == ["intercept", "position=BTN", "stack=<30", "stage=bubble"]
    assert X.tolist() == [
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0],
    ]


def test_levels_absent_from_data_are_dropped():
    X, names = encode([row(), row(zone="never")])
    assert names == ["intercept", "zone=never"]
    assert X.shape == (2, 2)


def test_first_present_level_becomes_reference():
    # UTG в данных нет — опорным становится первый присутствующий уровень (SB раньше BB).
    X, names = encode([row(position="BB"), row(position="SB")])
    assert names == ["intercept", "position=BB"]
    assert X.tolist() == [[1, 1], [1, 0]]


def test_feature_subset():
    X, names = encode([row(), row(position="BTN", stage="bubble")], features=("stage",))
    assert names == ["intercept", "stage=bubble"]


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="position"):
        encode([row(position="UTG2")])


def test_encode_is_deterministic():
    rows = [row(position=p) for p in ("BB", "SB", "CO", "BTN")]
    assert encode(rows)[1] == encode(list(reversed(rows)))[1]
    assert np.array_equal(encode(rows)[0], encode(rows)[0])
