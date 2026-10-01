"""Отчёт по прогону (spec §11): сводка узлов, раскрытие, β с выводами, сетка 13×13.

Отклонения по классам рук для сетки не хранятся — считаются на лету из решений
версии маппинга прогона (она зафиксирована в analysis_runs), так что воспроизводимы.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.analysis.pipeline import load_items
from core.db.models import AnalysisRun, ModelCoef, Node, NodeStat
from core.normalize.hand_class import RANKS
from core.stats.engine import HAND_GROUPS, ZONES, StatsConfig

# Вывод по знаку β для известных шагов дерева: (β > 0, β < 0).
_STEP_WORDING = {
    "continue_vs_fold": ("продолжаете чаще солвера", "сбрасываете чаще солвера"),
    "aggr_vs_call": ("рейзите чаще солвера", "коллируете там, где солвер рейзит"),
}


@dataclass
class NodeReport:
    node_id: int
    key: str
    action: str
    top: NodeStat
    zones: list[NodeStat]
    groups: list[NodeStat]
    is_leak: bool


@dataclass
class CoefRow:
    feature: str
    beta: float
    se: float
    ci_low: float
    ci_high: float
    significant: bool
    text: str


@dataclass
class Report:
    run: AnalysisRun
    stats: StatsConfig
    nodes: list[NodeReport] = field(default_factory=list)
    coefs: dict[str, list[CoefRow]] = field(default_factory=dict)


@dataclass
class HandCell:
    n: int = 0
    observed: float = 0.0
    expected: float = 0.0
    weight: float = 0.0

    @property
    def deviation(self) -> float:
        """Взвешенная разница частот «факт − GTO» для руки, в [−1, 1]."""
        return (self.observed - self.expected) / self.weight if self.weight else 0.0


def grid_layout() -> list[list[str]]:
    """Стандартная сетка: пары по диагонали, одномастные справа сверху, разномастные слева снизу."""
    grid = []
    for i, row_rank in enumerate(RANKS):
        row = []
        for j, col_rank in enumerate(RANKS):
            if i == j:
                row.append(row_rank + col_rank)
            elif i < j:
                row.append(f"{row_rank}{col_rank}s")
            else:
                row.append(f"{col_rank}{row_rank}o")
        grid.append(row)
    return grid


def describe_coef(step: str, feature: str, beta: float, ci_low: float, ci_high: float) -> str:
    subject = "На опорных уровнях признаков" if feature == "intercept" else f"{feature} (к опорному уровню)"
    if ci_low <= 0 <= ci_high:
        verdict = "нет значимого отклонения"
    else:
        more, less = _STEP_WORDING.get(step, (f"{step}: чаще солвера", f"{step}: реже солвера"))
        verdict = more if beta > 0 else less
    text = f"{subject}: {verdict} (шансы ×{math.exp(beta):.2f})"
    if feature.startswith("stage="):
        text += " — частично ожидаемо из-за ICM"
    return text


def load_report(session: Session, run_id: int) -> Report | None:
    run = session.get(AnalysisRun, run_id)
    if run is None:
        return None
    stats = StatsConfig(**run.params.get("stats", {}))
    report = Report(run=run, stats=stats)

    grouped: dict[tuple[int, str], dict] = defaultdict(lambda: {"top": None, "zones": [], "groups": []})
    keys = {}
    rows = session.execute(
        select(NodeStat, Node.key).join(Node, Node.id == NodeStat.node_id).where(NodeStat.run_id == run_id)
    ).all()
    for stat, key in rows:
        entry = grouped[(stat.node_id, stat.action)]
        keys[stat.node_id] = key
        if stat.zone:
            entry["zones"].append(stat)
        elif stat.hand_group:
            entry["groups"].append(stat)
        else:
            entry["top"] = stat

    for (node_id, action), entry in grouped.items():
        top = entry["top"]
        report.nodes.append(NodeReport(
            node_id=node_id,
            key=keys[node_id],
            action=action,
            top=top,
            zones=sorted(entry["zones"], key=lambda s: ZONES.index(s.zone)),
            groups=sorted(entry["groups"], key=lambda s: HAND_GROUPS.index(s.hand_group)),
            is_leak=top.n >= stats.min_sample and abs(top.z) > stats.z_threshold,
        ))
    report.nodes.sort(key=lambda r: (-abs(r.top.z), r.key, r.action))

    for coef in session.scalars(select(ModelCoef).where(ModelCoef.run_id == run_id).order_by(ModelCoef.id)):
        report.coefs.setdefault(coef.tree_step, []).append(CoefRow(
            feature=coef.feature,
            beta=coef.beta,
            se=coef.se,
            ci_low=coef.ci_low,
            ci_high=coef.ci_high,
            significant=not coef.ci_low <= 0 <= coef.ci_high,
            text=describe_coef(coef.tree_step, coef.feature, coef.beta, coef.ci_low, coef.ci_high),
        ))
    return report


def hand_deviations(session: Session, run_id: int, node_id: int, action: str) -> dict[str, HandCell] | None:
    """Факт и GTO-ожидание действия по каждому из 169 классов; None — нет прогона, узла или действия."""
    run = session.get(AnalysisRun, run_id)
    node = session.get(Node, node_id)
    if run is None or node is None:
        return None
    items = [i for i in load_items(session, node.chart_set_id, run.mapping_version) if i.node_id == node_id]
    if items and action not in items[0].node_actions:
        return None

    cells = {hand: HandCell() for row in grid_layout() for hand in row}
    for item in items:
        cell = cells[item.decision.hand_class]
        cell.n += 1
        cell.weight += item.weight
        cell.expected += item.weight * item.strategy.get(action, 0.0)
        cell.observed += item.weight * (item.action == action)
    return cells
