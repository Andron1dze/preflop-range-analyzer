"""Канонический формат GTO-чартов (spec §7).

Файл (JSON или YAML) — один набор чартов:

    chart_set: {model: chipev, ante: {type: bb, size_bb: 1.0}, source: ..., version: "1"}
    nodes:
      - key: "40bb|CO:open|HERO=BB"
        stack_bb: 40
        sizings: {CO: 2.2, BB: 9.5}     # bb; включая рейз самого героя
        strategy: {AKs: {call: 0.35, raise: 0.65}, ...}   # все 169 классов

Отсутствующее в стратегии действие — частота 0. Сумма по руке в пределах
допуска перенормируется ровно к 1. Все ошибки собираются и выдаются разом.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.model import POSITIONS_8MAX, AnteType
from core.normalize.hand_class import ALL_HAND_CLASSES

# Чарты для тестов: загружаются, но в пользовательский анализ не попадают.
SYNTHETIC_SOURCE = "synthetic-test"
DEFAULT_TOLERANCE = 0.005
ACTIONS = ("fold", "check", "call", "raise", "allin")

_MODEL_RE = re.compile(r"^(chipev|icm:.+)$")
_HERO_RE = re.compile(r"\|HERO=(?P<position>[A-Z0-9]+)$")
_LOADERS = {".json": json.loads, ".yaml": yaml.safe_load, ".yml": yaml.safe_load}


class ChartValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid chart set:\n" + "\n".join(f"- {e}" for e in errors))


@dataclass
class ChartNode:
    key: str
    hero_position: str
    stack_bb: float
    sizings: dict[str, float]
    # hand_class → {action: freq}, только ненулевые частоты, сумма = 1.
    strategy: dict[str, dict[str, float]]


@dataclass
class ChartSetSpec:
    model: str
    ante_type: str
    ante_size_bb: float
    source: str
    version: str
    nodes: list[ChartNode] = field(default_factory=list)

    @property
    def eligible_for_analysis(self) -> bool:
        return self.source != SYNTHETIC_SOURCE


def load_chart_file(path: str | Path, *, tolerance: float = DEFAULT_TOLERANCE) -> ChartSetSpec:
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ChartValidationError([f"{path.name}: unsupported extension {path.suffix!r}"])
    return parse_chart_set(loader(path.read_text(encoding="utf-8")), tolerance=tolerance)


def parse_chart_set(data: Any, *, tolerance: float = DEFAULT_TOLERANCE) -> ChartSetSpec:
    errors: list[str] = []
    if not isinstance(data, dict):
        raise ChartValidationError(["top level must be a mapping with chart_set and nodes"])

    header = data.get("chart_set")
    if not isinstance(header, dict):
        errors.append("chart_set: missing")
        header = {}
    for name in ("model", "source", "version"):
        if not isinstance(header.get(name), (str, int)) or str(header.get(name)) == "":
            errors.append(f"chart_set.{name}: missing")
    model = str(header.get("model", ""))
    if model and not _MODEL_RE.match(model):
        errors.append(f"chart_set.model: {model!r} is not 'chipev' or 'icm:<profile>'")
    ante_type, ante_size = _parse_ante(header.get("ante"), errors)

    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        errors.append("nodes: at least one node is required")
        raw_nodes = []
    nodes = []
    seen = set()
    for index, raw in enumerate(raw_nodes):
        node = _parse_node(raw, f"nodes[{index}]", tolerance, errors)
        if node is None:
            continue
        identity = (node.key, node.stack_bb, tuple(sorted(node.sizings.items())))
        if identity in seen:
            errors.append(f"nodes[{index}] {node.key}: duplicate node")
        seen.add(identity)
        nodes.append(node)

    if errors:
        raise ChartValidationError(errors)
    return ChartSetSpec(
        model=model,
        ante_type=ante_type,
        ante_size_bb=ante_size,
        source=str(header["source"]),
        version=str(header["version"]),
        nodes=nodes,
    )


def _parse_ante(raw: Any, errors: list[str]) -> tuple[str, float]:
    if not isinstance(raw, dict):
        errors.append("chart_set.ante: missing structure {type, size_bb}")
        return "", 0.0
    ante_type = raw.get("type")
    size = raw.get("size_bb")
    if ante_type not in {t.value for t in AnteType}:
        errors.append(f"chart_set.ante.type: {ante_type!r} is not one of none/each/bb")
    if not _is_number(size) or size < 0:
        errors.append(f"chart_set.ante.size_bb: {size!r} is not a non-negative number")
    elif ante_type == AnteType.NONE and size != 0:
        errors.append("chart_set.ante: size_bb must be 0 when type is none")
    return str(ante_type), float(size) if _is_number(size) else 0.0


def _parse_node(raw: Any, where: str, tolerance: float, errors: list[str]) -> ChartNode | None:
    if not isinstance(raw, dict):
        errors.append(f"{where}: node must be a mapping")
        return None
    key = raw.get("key")
    hero = _HERO_RE.search(key) if isinstance(key, str) else None
    if hero and hero["position"] in POSITIONS_8MAX:
        where = f"{where} {key}"
    else:
        errors.append(f"{where}: key {key!r} must end with |HERO=<position>")

    stack = raw.get("stack_bb")
    if not _is_number(stack) or stack <= 0:
        errors.append(f"{where}: stack_bb {stack!r} is not a positive number")

    sizings = raw.get("sizings")
    if not isinstance(sizings, dict) or not sizings:
        errors.append(f"{where}: sizings are required (position → size in bb)")
        sizings = {}
    for position, size in sizings.items():
        if position not in POSITIONS_8MAX:
            errors.append(f"{where}: sizing for unknown position {position!r}")
        if not _is_number(size) or size <= 0:
            errors.append(f"{where}: sizing for {position} {size!r} is not a positive number")

    strategy = _parse_strategy(raw.get("strategy"), where, tolerance, errors)
    hero_position = hero["position"] if hero else ""
    if hero and any("raise" in freqs for freqs in strategy.values()) and hero_position not in sizings:
        errors.append(f"{where}: strategy has raise but no sizing for hero position {hero_position}")

    if not hero or not _is_number(stack) or stack <= 0:
        return None
    return ChartNode(
        key=key,
        hero_position=hero_position,
        stack_bb=float(stack),
        sizings={p: float(s) for p, s in sizings.items() if _is_number(s)},
        strategy=strategy,
    )


def _parse_strategy(raw: Any, where: str, tolerance: float, errors: list[str]) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict):
        errors.append(f"{where}: strategy is required")
        return {}

    unknown = sorted(set(raw) - set(ALL_HAND_CLASSES))
    if unknown:
        errors.append(f"{where}: unknown hand classes {', '.join(map(str, unknown))}")
    missing = [h for h in ALL_HAND_CLASSES if h not in raw]
    if missing:
        errors.append(f"{where}: missing {len(missing)} of 169 hand classes: {', '.join(missing)}")

    strategy = {}
    for hand in ALL_HAND_CLASSES:
        freqs = raw.get(hand)
        if freqs is None:
            continue
        if not isinstance(freqs, dict):
            errors.append(f"{where} {hand}: expected mapping action → frequency")
            continue
        bad = False
        for action, freq in freqs.items():
            if action not in ACTIONS:
                errors.append(f"{where} {hand}: unknown action {action!r}")
                bad = True
            elif not _is_number(freq) or not 0 <= freq <= 1:
                errors.append(f"{where} {hand}: frequency of {action} {freq!r} is not in [0, 1]")
                bad = True
        if bad:
            continue
        total = sum(freqs.values())
        if not math.isclose(total, 1.0, abs_tol=tolerance):
            errors.append(f"{where} {hand}: frequencies sum to {total:g}, expected 1 ± {tolerance:g}")
            continue
        strategy[hand] = {a: f / total for a, f in freqs.items() if f > 0}
    return strategy


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
