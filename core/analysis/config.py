"""Параметры прогона анализа. Сохраняются в analysis_runs.params для воспроизводимости."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from core.models.action_tree import DEFAULT_TREE, TreeStep, load_tree
from core.stats.engine import StatsConfig

FEATURES = ("position", "stack", "facing", "hand_group", "zone", "stage")


@dataclass(frozen=True)
class AnalysisConfig:
    stats: StatsConfig = field(default_factory=StatsConfig)
    tree: tuple[TreeStep, ...] = DEFAULT_TREE
    l2: float = 1.0
    # Временные бакеты до решения по сетке узлов (тикет 15), в bb.
    stack_edges: tuple[float, ...] = (20.0, 30.0, 45.0)
    # Размер ставки, на которую отвечает игрок, в bb.
    facing_edges: tuple[float, ...] = (2.5, 3.5)
    features: tuple[str, ...] = FEATURES

    def __post_init__(self):
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")
        for name in ("stack_edges", "facing_edges"):
            edges = getattr(self, name)
            if not edges or any(e <= 0 for e in edges) or list(edges) != sorted(set(edges)):
                raise ValueError(f"{name} must be positive and strictly increasing")
        unknown = set(self.features) - set(FEATURES)
        if unknown or not self.features:
            raise ValueError(f"features must be a non-empty subset of {FEATURES}, got unknown {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": asdict(self.stats),
            "tree": {
                "steps": [
                    {
                        "name": step.name,
                        "population": None if step.population is None else sorted(step.population),
                        "positive": sorted(step.positive),
                    }
                    for step in self.tree
                ]
            },
            "l2": self.l2,
            "stack_edges": list(self.stack_edges),
            "facing_edges": list(self.facing_edges),
            "features": list(self.features),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AnalysisConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"unknown analysis settings: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        if "stats" in mapping:
            try:
                kwargs["stats"] = StatsConfig(**mapping["stats"])
            except TypeError as error:
                raise ValueError(f"stats: {error}") from error
        if "tree" in mapping:
            kwargs["tree"] = load_tree(mapping["tree"])
        if "l2" in mapping:
            kwargs["l2"] = float(mapping["l2"])
        for name in ("stack_edges", "facing_edges"):
            if name in mapping:
                kwargs[name] = tuple(float(x) for x in mapping[name])
        if "features" in mapping:
            kwargs["features"] = tuple(mapping["features"])
        return cls(**kwargs)


def load_config(path: str | Path) -> AnalysisConfig:
    return AnalysisConfig.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
