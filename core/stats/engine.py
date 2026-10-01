"""Статистический движок (spec §8).

Проверяется одно бинарное событие (например, «сбросил»): pᵢ — GTO-частота этого
действия для реально пришедшей руки, xᵢ — сделал ли его игрок, wᵢ — вес по
расстоянию до узла. Факт и ожидание сравниваются через Пуассон–Бернулли:

    E = Σ wᵢ·pᵢ,   Var = Σ wᵢ²·pᵢ(1−pᵢ),   z = (Σ wᵢ·xᵢ − E) / √Var

Частоты ограничиваются [ε, 1−ε]: в чартах 0 и 1 обычно означают округление,
а действие с частотой ровно 0 давало бы z = ±∞ от единственного мисклика.
Зоны частот считаются по исходным, неограниченным частотам.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from core.normalize.hand_class import RANKS

DEFAULT_EPSILON = 1e-4


@dataclass(frozen=True)
class StatsConfig:
    z_threshold: float = 2.0
    min_sample: int = 30
    # Зоны GTO-частоты: p > pure — «чистое действие», p < never — «никогда», иначе «микс».
    pure_threshold: float = 0.9
    never_threshold: float = 0.1
    # Ограничение частот [ε, 1−ε] при расчёте z.
    p_epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True)
class DeviationTest:
    n: int
    expected: float
    variance: float
    observed: float
    z: float


@dataclass(frozen=True)
class Observation:
    hand_class: str
    p: float
    did: bool
    w: float = 1.0
    stage_known: bool = True


@dataclass(frozen=True)
class NodeStatRow:
    """Строка отчёта по узлу: zone и hand_group пусты на верхнем уровне."""

    zone: str | None
    hand_group: str | None
    n: int
    expected: float
    observed: float
    z: float
    unknown_stage_share: float
    is_leak: bool


def poisson_binomial_test(
    p: Iterable[float],
    did: Iterable[bool],
    w: Iterable[float] | None = None,
    *,
    eps: float = DEFAULT_EPSILON,
) -> DeviationTest:
    p, did = list(p), [bool(x) for x in did]
    w = [1.0] * len(p) if w is None else list(w)
    if not len(p) == len(did) == len(w):
        raise ValueError("p, did and w must have the same length")
    if any(not 0 <= pi <= 1 for pi in p):
        raise ValueError("p must be in [0, 1]")
    if any(wi < 0 for wi in w):
        raise ValueError("weights must be non-negative")
    p = [min(max(pi, eps), 1 - eps) for pi in p]

    expected = sum(wi * pi for wi, pi in zip(w, p))
    variance = sum(wi * wi * pi * (1 - pi) for wi, pi in zip(w, p))
    observed = sum(wi for wi, xi in zip(w, did) if xi)
    diff = observed - expected
    if variance > 0:
        z = diff / math.sqrt(variance)
    elif math.isclose(diff, 0, abs_tol=1e-12):
        z = 0.0
    else:
        z = math.copysign(math.inf, diff)
    return DeviationTest(n=len(p), expected=expected, variance=variance, observed=observed, z=z)


def two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def bh_adjust(pvalues: Sequence[float]) -> list[float]:
    """q-значения Benjamini–Hochberg в исходном порядке."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    qvalues = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        qvalues[i] = running
    return qvalues


def zone_of(p: float, config: StatsConfig) -> str:
    if p > config.pure_threshold:
        return "pure"
    if p < config.never_threshold:
        return "never"
    return "mix"


def hand_group(hand: str) -> str:
    high, low = hand[0], hand[1]
    if high == low:
        return "pair"
    suited = hand.endswith("s")
    prefix = "suited" if suited else "offsuit"
    if RANKS.index(low) <= RANKS.index("T"):
        return f"{prefix}_broadway"
    if high == "A":
        return f"{prefix}_ace"
    # Коннекторы и одногэпперы: 98s, 97s.
    if suited and RANKS.index(low) - RANKS.index(high) <= 2:
        return "suited_connector"
    return f"{prefix}_other"


def analyze_node(observations: Sequence[Observation], config: StatsConfig) -> list[NodeStatRow]:
    """Верхний уровень узла, затем разбивка по зонам частоты и по группам рук."""
    rows = [_row(observations, None, None, config)]
    for zone in ("pure", "mix", "never"):
        subset = [o for o in observations if zone_of(o.p, config) == zone]
        if subset:
            rows.append(_row(subset, zone, None, config))
    groups = sorted({hand_group(o.hand_class) for o in observations})
    for group in groups:
        rows.append(_row([o for o in observations if hand_group(o.hand_class) == group], None, group, config))
    return rows


def _row(subset: Sequence[Observation], zone: str | None, group: str | None, config: StatsConfig) -> NodeStatRow:
    test = poisson_binomial_test(
        [o.p for o in subset], [o.did for o in subset], [o.w for o in subset], eps=config.p_epsilon
    )
    unknown = sum(1 for o in subset if not o.stage_known)
    return NodeStatRow(
        zone=zone,
        hand_group=group,
        n=test.n,
        expected=test.expected,
        observed=test.observed,
        z=test.z,
        unknown_stage_share=unknown / test.n if test.n else 0.0,
        is_leak=test.n >= config.min_sample and abs(test.z) > config.z_threshold,
    )
