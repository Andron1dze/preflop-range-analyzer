"""Таблицы SQLite (spec §10).

Точная линия решения (`decisions`) и её привязка к узлу (`decision_node_map`)
хранятся раздельно: при смене сетки узлов пересчитывается только маппинг.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
)
from sqlalchemy.orm import Mapped, mapped_column

from core.db.base import Base

HAND_SOURCES = ("pokerstars", "manual")
ANTE_TYPES = ("none", "each", "bb")
STAGES = ("early", "mid", "bubble", "itm", "ft")
STAGE_SOURCES = ("proxy", "summary", "manual")
RUN_STATUSES = ("pending", "running", "done", "failed")


def _one_of(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Hand(Base):
    __tablename__ = "hands"
    __table_args__ = (
        CheckConstraint(_one_of("source", HAND_SOURCES), name="source"),
        CheckConstraint(_one_of("ante_type", ANTE_TYPES), name="ante_type"),
        # Повторный импорт той же раздачи не создаёт дубликат.
        UniqueConstraint("source", "external_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    # Номер раздачи в источнике (у PokerStars — "Hand #..."); у ручного ввода может отсутствовать.
    external_id: Mapped[str | None] = mapped_column(String(64))
    # Сырьё для ре-парсинга: текст HH или сериализованная форма ручного ввода.
    raw_text: Mapped[str] = mapped_column(Text)
    tournament_id: Mapped[str | None] = mapped_column(String(64))
    level: Mapped[int | None] = mapped_column(Integer)
    # Анте в bb и кто его платит; тип пуст у раздач, импортированных до его появления.
    ante: Mapped[float | None] = mapped_column(Float)
    ante_type: Mapped[str | None] = mapped_column(String(8))
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Decision(Base):
    __tablename__ = "decisions"
    __table_args__ = (
        CheckConstraint(_one_of("stage", STAGES), name="stage"),
        CheckConstraint(_one_of("stage_source", STAGE_SOURCES), name="stage_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    hand_id: Mapped[int] = mapped_column(ForeignKey("hands.id", ondelete="CASCADE"), index=True)
    # "hero" или идентификатор оппонента: ядро сравнения не привязано к игроку.
    actor: Mapped[str] = mapped_column(String(64), index=True)
    position: Mapped[str] = mapped_column(String(8))
    # Карты оппонентов обычно неизвестны.
    hole_cards: Mapped[str | None] = mapped_column(String(4))
    hand_class: Mapped[str | None] = mapped_column(String(3))
    action: Mapped[str] = mapped_column(String(16))
    size_bb: Mapped[float | None] = mapped_column(Float)
    size_pot: Mapped[float | None] = mapped_column(Float)
    eff_stack_bb: Mapped[float] = mapped_column(Float)
    # Олл-ин: рейз на весь стек сопоставляется с действием "allin" чарта.
    all_in: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    # Нормализованная точная линия до решения.
    line: Mapped[str] = mapped_column(Text)
    stage: Mapped[str | None] = mapped_column(String(8))
    stage_source: Mapped[str | None] = mapped_column(String(8))
    # Заполняется, только если у чартов есть EV действий.
    ev_loss: Mapped[float | None] = mapped_column(Float)


class ChartSet(Base):
    __tablename__ = "chart_sets"
    __table_args__ = (
        # GLOB чувствителен к регистру и требует непустой профиль после "icm:".
        CheckConstraint("model = 'chipev' OR model GLOB 'icm:?*'", name="model"),
        UniqueConstraint("source", "version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column(String(64))
    ante: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32))
    # False для синтетических чартов: загружаются для тестов, в анализ не попадают.
    eligible_for_analysis: Mapped[bool] = mapped_column(Boolean)


class Node(Base):
    __tablename__ = "nodes"
    __table_args__ = (UniqueConstraint("chart_set_id", "key", "stack_bucket", "sizing_bucket"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    chart_set_id: Mapped[int] = mapped_column(ForeignKey("chart_sets.id", ondelete="CASCADE"))
    # Типы действий и позиции, например "40bb|CO:open|HERO=BTN". Бакеты — отдельными полями.
    key: Mapped[str] = mapped_column(String(256))
    stack_bucket: Mapped[str] = mapped_column(String(32))
    sizing_bucket: Mapped[str] = mapped_column(String(32))
    # Реальные сайзинги узла по позициям, например {"CO": 2.2}.
    sizings: Mapped[dict[str, Any]] = mapped_column(JSON)


class NodeStrategy(Base):
    __tablename__ = "node_strategies"
    __table_args__ = (CheckConstraint("freq >= 0 AND freq <= 1", name="freq"),)

    node_id: Mapped[int] = mapped_column(
        ForeignKey("nodes.id", ondelete="CASCADE"), primary_key=True
    )
    hand_class: Mapped[str] = mapped_column(String(3), primary_key=True)
    action: Mapped[str] = mapped_column(String(16), primary_key=True)
    freq: Mapped[float] = mapped_column(Float)


class DecisionNodeMap(Base):
    __tablename__ = "decision_node_map"

    decision_id: Mapped[int] = mapped_column(
        ForeignKey("decisions.id", ondelete="CASCADE"), primary_key=True
    )
    mapping_version: Mapped[str] = mapped_column(String(32), primary_key=True)
    node_id: Mapped[int] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), index=True)
    distance: Mapped[float] = mapped_column(Float)
    weight: Mapped[float] = mapped_column(Float)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (CheckConstraint(_one_of("status", RUN_STATUSES), name="status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    mapping_version: Mapped[str] = mapped_column(String(32))
    chart_set_version: Mapped[str] = mapped_column(String(32))
    # Пороги и прочие параметры прогона: нужны для воспроизводимости отчёта.
    params: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="pending")


class NodeStat(Base):
    """Строка отчёта: zone и hand_group пусты на верхнем уровне узла."""

    __tablename__ = "node_stats"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    node_id: Mapped[int] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"))
    # Проверяемое действие узла: движок тестирует одно действие за раз.
    action: Mapped[str] = mapped_column(String(16))
    zone: Mapped[str | None] = mapped_column(String(16))
    hand_group: Mapped[str | None] = mapped_column(String(32))
    n: Mapped[int] = mapped_column(Integer)
    expected: Mapped[float] = mapped_column(Float)
    observed: Mapped[float] = mapped_column(Float)
    z: Mapped[float] = mapped_column(Float)
    bh_q: Mapped[float | None] = mapped_column(Float)
    unknown_stage_share: Mapped[float] = mapped_column(Float)


class ModelCoef(Base):
    __tablename__ = "model_coefs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    # Шаг дерева действий: "continue_vs_fold", "aggr_vs_call", ...
    tree_step: Mapped[str] = mapped_column(String(32))
    feature: Mapped[str] = mapped_column(String(64))
    beta: Mapped[float] = mapped_column(Float)
    se: Mapped[float] = mapped_column(Float)
    ci_low: Mapped[float] = mapped_column(Float)
    ci_high: Mapped[float] = mapped_column(Float)
