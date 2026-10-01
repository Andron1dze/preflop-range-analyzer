"""Расстояние от точной линии до узла и вес решения (тикет 15).

    d_stack = ln(eff / S_узла) / σ
    d_size  = ln(факт / узел) / σ   — для каждого рейза до решения (jam против jam — 0)
    d = √(d_stack² + Σ d_size²),   w = exp(−d²/2),   w < cutoff → 0

Текущий рейз героя в линию не входит и на расстояние не влияет. Формат чарта
хранит один сайзинг на позицию, поэтому сравнивается последний рейз позиции.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from core.mapping.line import parse_line


@dataclass(frozen=True)
class MappingConfig:
    name: str = "grid-v1"
    sigma: float = math.log(1.25)
    cutoff: float = 0.1

    def version(self, chart_set_id: int) -> str:
        return f"{self.name}/cs{chart_set_id}/sigma={self.sigma:.6g}/cutoff={self.cutoff:g}"


@dataclass(frozen=True)
class NodeCandidate:
    node_id: int
    stack_bb: float
    sizings: dict[str, float]


@dataclass(frozen=True)
class Match:
    node_id: int
    distance: float
    weight: float


def stack_distance(eff_stack_bb: float, node_stack_bb: float, config: MappingConfig) -> float:
    return math.log(eff_stack_bb / node_stack_bb) / config.sigma


def size_distance(actual_bb: float, node_bb: float, config: MappingConfig) -> float:
    return math.log(actual_bb / node_bb) / config.sigma


def weight(distance: float, config: MappingConfig) -> float:
    w = math.exp(-distance * distance / 2)
    return w if w >= config.cutoff else 0.0


def match_decision(
    line: str, eff_stack_bb: float, candidates: Sequence[NodeCandidate], config: MappingConfig
) -> Match | None:
    """Ближайший узел среди кандидатов с тем же ключом; None, если ни один не подходит."""
    last_raise = {t.position: t for t in parse_line(line) if t.code == "R"}
    best = None
    for node in candidates:
        squares = stack_distance(eff_stack_bb, node.stack_bb, config) ** 2
        for position, token in last_raise.items():
            if token.all_in:
                continue
            node_size = node.sizings.get(position)
            if node_size is None:
                break
            squares += size_distance(token.size_bb, node_size, config) ** 2
        else:
            distance = math.sqrt(squares)
            if best is None or distance < best.distance:
                best = Match(node.node_id, distance, weight(distance, config))
    return best
