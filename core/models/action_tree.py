"""Дерево действий: выбор в узле раскладывается на бинарные шаги (spec §9).

Шаг задаётся популяцией (какие решения в него входят; None — все) и
положительными действиями. Смещение шага — logit условной GTO-частоты:

    p = Σ p(положительные) / Σ p(популяция),   обрезанная до [ε, 1−ε]

Строка исключается, если условная частота не определена (Σ p(популяция) < ε)
или шаг в узле вырожден: у узла нет действий одной из сторон (в RFI нет колла).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.charts.format import ACTIONS
from core.models.offset_logit import logit

_CONTINUE = frozenset({"check", "call", "raise", "allin"})


@dataclass(frozen=True)
class TreeStep:
    name: str
    population: frozenset[str] | None
    positive: frozenset[str]


DEFAULT_TREE = (
    TreeStep("continue_vs_fold", None, _CONTINUE),
    TreeStep("aggr_vs_call", _CONTINUE, frozenset({"raise", "allin"})),
)


@dataclass(frozen=True)
class StepInput:
    # GTO-стратегия для пришедшей руки: действие → частота.
    strategy: Mapping[str, float]
    action: str
    # Действия, которые вообще есть в узле.
    node_actions: frozenset[str]


@dataclass
class StepData:
    # Индексы входных решений, попавших в шаг.
    rows: list[int]
    y: np.ndarray
    offset: np.ndarray


def load_tree(config: Mapping[str, Any]) -> tuple[TreeStep, ...]:
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("action tree: at least one step is required")
    tree = []
    for index, raw in enumerate(steps):
        name = raw.get("name")
        if not name:
            raise ValueError(f"action tree step {index}: name is required")
        population = None if raw.get("population") is None else frozenset(raw["population"])
        positive = frozenset(raw.get("positive") or ())
        unknown = (positive | (population or frozenset())) - set(ACTIONS)
        if unknown:
            raise ValueError(f"action tree step {name}: unknown actions {sorted(unknown)}")
        if not positive:
            raise ValueError(f"action tree step {name}: positive actions are required")
        if population is not None and not positive <= population:
            raise ValueError(f"action tree step {name}: positive actions must be in population")
        tree.append(TreeStep(name, population, positive))
    return tuple(tree)


def step_dataset(step: TreeStep, decisions: Sequence[StepInput], *, eps: float) -> StepData:
    rows, y, p = [], [], []
    for index, decision in enumerate(decisions):
        population = step.population
        if population is not None and decision.action not in population:
            continue
        available = decision.node_actions if population is None else decision.node_actions & population
        if not available & step.positive or not available - step.positive:
            continue
        denominator = 1.0 if population is None else sum(decision.strategy.get(a, 0.0) for a in population)
        if denominator < eps:
            continue
        numerator = sum(decision.strategy.get(a, 0.0) for a in step.positive)
        rows.append(index)
        y.append(1.0 if decision.action in step.positive else 0.0)
        p.append(min(max(numerator / denominator, eps), 1 - eps))
    return StepData(rows=rows, y=np.array(y), offset=logit(np.array(p)) if p else np.array([]))
