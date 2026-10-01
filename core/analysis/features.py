"""Признаки регрессии: категориальные значения решения → матрица X (one-hot).

Опорный уровень признака — первый присутствующий в данных уровень в порядке
FEATURE_LEVELS; поэтому порядок задаёт «естественную» точку отсчёта: UTG,
номинальный стек 40bb, «нет рейза», пары, зона «микс», неизвестная стадия.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from core.analysis.config import FEATURES, AnalysisConfig
from core.db.models import STAGES
from core.mapping.line import facing_action
from core.model import POSITIONS_8MAX
from core.stats.engine import HAND_GROUPS, hand_group, zone_of

NOMINAL_STACK_BB = 40.0


def bucket_labels(edges: Sequence[float]) -> list[str]:
    labels = [f"<{edges[0]:g}"]
    labels += [f"{lo:g}-{hi:g}" for lo, hi in zip(edges, edges[1:])]
    labels.append(f"{edges[-1]:g}+")
    return labels


def _bucket(value: float, edges: Sequence[float]) -> str:
    labels = bucket_labels(edges)
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def stack_bucket(eff_stack_bb: float, edges: Sequence[float]) -> str:
    return _bucket(eff_stack_bb, edges)


def translate_action(action: str, all_in: bool, node_actions: frozenset[str]) -> str | None:
    """Действие решения → действие чарта; None, если в узле такого действия нет."""
    if action != "raise":
        return action if action in node_actions else None
    preferred, fallback = ("allin", "raise") if all_in else ("raise", "allin")
    for candidate in (preferred, fallback):
        if candidate in node_actions:
            return candidate
    return None


def feature_levels(config: AnalysisConfig) -> dict[str, list[str]]:
    stacks = bucket_labels(config.stack_edges)
    nominal = stack_bucket(NOMINAL_STACK_BB, config.stack_edges)
    return {
        "position": list(POSITIONS_8MAX),
        "stack": [nominal] + [s for s in stacks if s != nominal],
        "facing": ["none", "open", "3bet", "4bet", "jam"],
        "hand_group": list(HAND_GROUPS),
        "zone": ["mix", "pure", "never"],
        "stage": ["unknown", *STAGES],
    }


FEATURE_LEVELS = feature_levels(AnalysisConfig())


def feature_values(
    *,
    position: str,
    eff_stack_bb: float,
    line: str,
    hand_class: str,
    p_step: float,
    stage: str | None,
    config: AnalysisConfig,
) -> dict[str, str]:
    return {
        "position": position,
        "stack": stack_bucket(eff_stack_bb, config.stack_edges),
        "facing": facing_action(line),
        "hand_group": hand_group(hand_class),
        "zone": zone_of(p_step, config.stats),
        "stage": stage or "unknown",
    }


def encode(
    rows: Sequence[Mapping[str, str]],
    levels: Mapping[str, Sequence[str]] = FEATURE_LEVELS,
    features: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    features = list(features or FEATURES)
    names = ["intercept"]
    columns = [np.ones(len(rows))]
    for feature in features:
        values = [row[feature] for row in rows]
        unknown = set(values) - set(levels[feature])
        if unknown:
            raise ValueError(f"{feature}: unknown levels {sorted(unknown)}")
        present = [level for level in levels[feature] if level in set(values)]
        for level in present[1:]:
            names.append(f"{feature}={level}")
            columns.append(np.array([1.0 if v == level else 0.0 for v in values]))
    return np.column_stack(columns), names
