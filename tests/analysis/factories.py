"""Синтетические данные для прогона анализа. Частоты выдуманы и к GTO отношения не имеют."""

import numpy as np

from core.charts.format import SYNTHETIC_SOURCE, parse_chart_set
from core.charts.store import save_chart_set
from core.db.models import Decision, DecisionNodeMap, Hand, Node
from core.normalize.hand_class import ALL_HAND_CLASSES

# BTN против открытия CO: fold 0.5 / call 0.3 / raise 0.2 у каждой руки.
VS_OPEN_KEY = "40bb|CO:open|HERO=BTN"
VS_OPEN = {"fold": 0.5, "call": 0.3, "raise": 0.2}
# Открытие с CO: raise 0.3 / fold 0.7.
RFI_KEY = "40bb|HERO=CO"
RFI = {"fold": 0.7, "raise": 0.3}


def chart_set_data():
    return {
        "chart_set": {
            "model": "chipev",
            "ante": {"type": "bb", "size_bb": 1.0},
            "source": SYNTHETIC_SOURCE,
            "version": "1",
        },
        "nodes": [
            {
                "key": VS_OPEN_KEY,
                "stack_bb": 40,
                "sizings": {"CO": 2.2, "BTN": 7.0},
                "strategy": {h: dict(VS_OPEN) for h in ALL_HAND_CLASSES},
            },
            {
                "key": RFI_KEY,
                "stack_bb": 40,
                "sizings": {"CO": 2.2},
                "strategy": {h: dict(RFI) for h in ALL_HAND_CLASSES},
            },
        ],
    }


def load_chart_set(session):
    chart_set = save_chart_set(session, parse_chart_set(chart_set_data()))
    session.flush()
    nodes = {n.key: n for n in session.query(Node).filter_by(chart_set_id=chart_set.id)}
    return chart_set, nodes


def add_decisions(
    session,
    node,
    actions,
    *,
    position,
    line,
    mapping_version="m1",
    actor="hero",
    weight=1.0,
    stage="mid",
    eff_stack_bb=40.0,
    seed=0,
):
    """Создаёт по раздаче на решение и привязывает решения к узлу."""
    rng = np.random.default_rng(seed)
    hands = rng.choice(ALL_HAND_CLASSES, size=len(actions))
    decisions = []
    for i, (action, hand_class) in enumerate(zip(actions, hands)):
        hand = Hand(source="manual", raw_text="{}", external_id=None)
        session.add(hand)
        session.flush()
        decision = Decision(
            hand_id=hand.id,
            actor=actor,
            position=position,
            hole_cards=None,
            hand_class=str(hand_class),
            action=action,
            size_bb=None,
            size_pot=None,
            eff_stack_bb=eff_stack_bb,
            all_in=False,
            line=line,
            stage=stage,
            stage_source="manual" if stage else None,
        )
        session.add(decision)
        decisions.append(decision)
    session.flush()
    for decision in decisions:
        session.add(DecisionNodeMap(
            decision_id=decision.id,
            mapping_version=mapping_version,
            node_id=node.id,
            distance=0.0,
            weight=weight,
        ))
    session.flush()
    return decisions


def sample_actions(probabilities, n, seed):
    rng = np.random.default_rng(seed)
    actions = list(probabilities)
    return list(rng.choice(actions, size=n, p=[probabilities[a] for a in actions]))
