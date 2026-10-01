"""Прогон анализа (spec §8–§10).

Берёт решения героя, привязанные к узлам выбранного набора чартов в заданной
версии маппинга (вес > 0), и:
1. по каждому узлу и каждому его действию считает статистику Пуассона–Бернулли
   (верхний уровень, зоны, группы рук) → node_stats; BH-поправка — по строкам
   верхнего уровня всего прогона;
2. по каждому шагу дерева действий обучает регрессию со смещением → model_coefs.

Результаты пишутся в savepoint: упавший прогон не оставляет частичных строк,
а сам помечается как failed. Коммит — на вызывающем.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.analysis.config import AnalysisConfig
from core.analysis.features import encode, feature_levels, feature_values, translate_action
from core.db.models import AnalysisRun, ChartSet, Decision, DecisionNodeMap, Node, NodeStat, NodeStrategy
from core.models.action_tree import StepInput, step_dataset
from core.models.offset_logit import fit_irls, save_coefs
from core.stats.engine import Observation, analyze_node, bh_adjust, two_sided_p


@dataclass(frozen=True)
class _Item:
    decision: Decision
    node_id: int
    weight: float
    strategy: dict[str, float]
    node_actions: frozenset[str]
    action: str


def run_analysis(
    session: Session,
    *,
    chart_set_id: int,
    mapping_version: str,
    config: AnalysisConfig,
    allow_synthetic: bool = False,
) -> AnalysisRun:
    chart_set = session.get(ChartSet, chart_set_id)
    if chart_set is None:
        raise ValueError(f"chart set {chart_set_id} not found")
    if not chart_set.eligible_for_analysis and not allow_synthetic:
        raise ValueError(f"chart set {chart_set_id} ({chart_set.source}) is not eligible for analysis")

    run = AnalysisRun(
        mapping_version=mapping_version,
        chart_set_version=f"{chart_set.source}@{chart_set.version}",
        params={**config.to_dict(), "allow_synthetic": allow_synthetic},
        status="running",
    )
    session.add(run)
    session.flush()
    try:
        with session.begin_nested():
            items = load_items(session, chart_set_id, mapping_version)
            _node_stats(session, run.id, items, config)
            _regression(session, run.id, items, config)
    except Exception:
        run.status = "failed"
        session.flush()
        raise
    run.status = "done"
    session.flush()
    return run


def load_items(session: Session, chart_set_id: int, mapping_version: str) -> list[_Item]:
    """Решения героя версии маппинга, привязанные к узлам набора, в словаре действий чарта."""
    rows = session.execute(
        select(Decision, DecisionNodeMap.node_id, DecisionNodeMap.weight)
        .join(DecisionNodeMap, DecisionNodeMap.decision_id == Decision.id)
        .join(Node, Node.id == DecisionNodeMap.node_id)
        .where(
            Decision.actor == "hero",
            Decision.hand_class.is_not(None),
            DecisionNodeMap.mapping_version == mapping_version,
            DecisionNodeMap.weight > 0,
            Node.chart_set_id == chart_set_id,
        )
        .order_by(Decision.id)
    ).all()

    node_ids = {node_id for _, node_id, _ in rows}
    strategies: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for s in session.scalars(select(NodeStrategy).where(NodeStrategy.node_id.in_(node_ids))):
        strategies[s.node_id][s.hand_class][s.action] = s.freq
    node_actions = {
        node_id: frozenset(a for freqs in hands.values() for a in freqs) for node_id, hands in strategies.items()
    }

    items = []
    for decision, node_id, weight in rows:
        actions = node_actions.get(node_id, frozenset())
        action = translate_action(decision.action, decision.all_in, actions)
        if action is None:
            continue
        items.append(_Item(
            decision=decision,
            node_id=node_id,
            weight=weight,
            strategy=dict(strategies[node_id][decision.hand_class]),
            node_actions=actions,
            action=action,
        ))
    return items


def _node_stats(session: Session, run_id: int, items: list[_Item], config: AnalysisConfig) -> None:
    by_node: dict[int, list[_Item]] = defaultdict(list)
    for item in items:
        by_node[item.node_id].append(item)

    top_level = []
    for node_id in sorted(by_node):
        node_items = by_node[node_id]
        for action in sorted(node_items[0].node_actions):
            observations = [
                Observation(
                    hand_class=item.decision.hand_class,
                    p=item.strategy.get(action, 0.0),
                    did=item.action == action,
                    w=item.weight,
                    stage_known=item.decision.stage is not None,
                )
                for item in node_items
            ]
            for row in analyze_node(observations, config.stats):
                stat = NodeStat(
                    run_id=run_id,
                    node_id=node_id,
                    action=action,
                    zone=row.zone,
                    hand_group=row.hand_group,
                    n=row.n,
                    expected=row.expected,
                    observed=row.observed,
                    z=row.z,
                    bh_q=None,
                    unknown_stage_share=row.unknown_stage_share,
                )
                session.add(stat)
                if row.zone is None and row.hand_group is None:
                    top_level.append(stat)

    for stat, q in zip(top_level, bh_adjust([two_sided_p(s.z) for s in top_level])):
        stat.bh_q = q
    session.flush()


def _regression(session: Session, run_id: int, items: list[_Item], config: AnalysisConfig) -> None:
    levels = feature_levels(config)
    inputs = [StepInput(item.strategy, item.action, item.node_actions) for item in items]
    for step in config.tree:
        data = step_dataset(step, inputs, eps=config.stats.p_epsilon)
        if not data.rows:
            continue
        p_step = 1 / (1 + np.exp(-data.offset))
        values = [
            feature_values(
                position=items[i].decision.position,
                eff_stack_bb=items[i].decision.eff_stack_bb,
                line=items[i].decision.line,
                hand_class=items[i].decision.hand_class,
                p_step=float(p),
                stage=items[i].decision.stage,
                config=config,
            )
            for i, p in zip(data.rows, p_step)
        ]
        X, names = encode(values, levels, config.features)
        if len(data.rows) <= X.shape[1]:
            continue
        weights = np.array([items[i].weight for i in data.rows])
        fit = fit_irls(X, data.y, data.offset, weights, l2=config.l2, unpenalized=[0])
        save_coefs(session, run_id, step.name, names, fit)
    session.flush()
