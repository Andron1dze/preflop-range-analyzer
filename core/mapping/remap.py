"""Пересчёт decision_node_map для набора чартов без перепарсинга раздач."""

import re
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core.db.models import ChartSet, Decision, DecisionNodeMap, Hand, Node
from core.mapping.distance import MappingConfig, NodeCandidate, match_decision
from core.mapping.line import abstract_line, split_node_key

_STACK_RE = re.compile(r"^([\d.]+)bb$")


@dataclass
class RemapReport:
    version: str
    # Привязаны с весом > 0.
    mapped: int = 0
    # Привязаны, но дальше порога: weight = 0, distance сохранён.
    far: int = 0
    # Узла с таким ключом нет (или у узла нет сайзинга для рейза линии).
    unmapped: int = 0


def remap(session: Session, chart_set_id: int, config: MappingConfig) -> RemapReport:
    """Перезаписывает привязки всех решений к узлам набора в версии `config.version(...)`."""
    if session.get(ChartSet, chart_set_id) is None:
        raise ValueError(f"chart set {chart_set_id} not found")
    report = RemapReport(version=config.version(chart_set_id))

    candidates: dict[tuple[str, str], list[NodeCandidate]] = defaultdict(list)
    for node in session.scalars(select(Node).where(Node.chart_set_id == chart_set_id).order_by(Node.id)):
        _, actions, hero = split_node_key(node.key)
        stack = _STACK_RE.match(node.stack_bucket)
        if stack is None:
            raise ValueError(f"node {node.key}: stack bucket {node.stack_bucket!r} is not '<N>bb'")
        candidates[(actions, hero)].append(NodeCandidate(node.id, float(stack[1]), dict(node.sizings)))

    session.execute(delete(DecisionNodeMap).where(DecisionNodeMap.mapping_version == report.version))
    rows = session.execute(
        select(Decision.id, Decision.position, Decision.line, Decision.eff_stack_bb, Hand.ante, Hand.ante_type)
        .join(Hand, Hand.id == Decision.hand_id)
        .order_by(Decision.id)
    )
    for decision_id, position, line, eff_stack_bb, ante, ante_type in rows:
        # Соглашение чарта: стеки до анте. При анте с каждого вычтено одинаково у всех —
        # прибавка точная; при BB-ante разница ≤ 1bb, её не учитываем.
        eff = eff_stack_bb + (ante or 0.0) * (ante_type == "each")
        match = match_decision(line, eff, candidates.get((abstract_line(line), position), []), config)
        if match is None:
            report.unmapped += 1
            continue
        session.add(DecisionNodeMap(
            decision_id=decision_id,
            mapping_version=report.version,
            node_id=match.node_id,
            distance=match.distance,
            weight=match.weight,
        ))
        if match.weight > 0:
            report.mapped += 1
        else:
            report.far += 1
    session.flush()
    return report
