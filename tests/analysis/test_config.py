import pytest
import yaml

from core.analysis.config import AnalysisConfig, load_config
from core.models.action_tree import DEFAULT_TREE
from core.stats.engine import StatsConfig


def test_defaults():
    config = AnalysisConfig()
    assert config.stats == StatsConfig()
    assert config.tree == DEFAULT_TREE
    assert config.l2 == 1.0
    assert config.stack_edges == (30.0, 35.0, 45.0)
    assert not hasattr(config, "facing_edges")  # ставка, на которую отвечаем, — тип действия
    assert config.features == ("position", "stack", "facing", "hand_group", "zone", "stage")


def test_round_trip_through_mapping():
    config = AnalysisConfig(stats=StatsConfig(z_threshold=2.5, min_sample=50), l2=0.5, features=("position",))
    assert AnalysisConfig.from_mapping(config.to_dict()) == config


def test_load_yaml(tmp_path):
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump({
        "stats": {"z_threshold": 3.0},
        "l2": 2.0,
        "tree": {"steps": [{"name": "continue_vs_fold", "population": None, "positive": ["call", "raise"]}]},
    }), encoding="utf-8")
    config = load_config(path)
    assert config.stats.z_threshold == 3.0
    assert config.stats.min_sample == 30  # остальное — по умолчанию
    assert config.l2 == 2.0
    assert [s.name for s in config.tree] == ["continue_vs_fold"]


@pytest.mark.parametrize(
    "mapping",
    [
        {"unknown": 1},
        {"stats": {"unknown": 1}},
        {"l2": -1},
        {"stack_edges": [30, 20]},
        {"features": ["position", "eye_color"]},
    ],
)
def test_invalid_config_is_rejected(mapping):
    with pytest.raises(ValueError):
        AnalysisConfig.from_mapping(mapping)
