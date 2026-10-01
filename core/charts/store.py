from sqlalchemy import select
from sqlalchemy.orm import Session

from core.charts.format import ChartSetSpec
from core.db import models


def save_chart_set(session: Session, chart: ChartSetSpec) -> models.ChartSet:
    """Записывает провалидированный набор чартов. Коммит — на вызывающем."""
    exists = session.scalar(
        select(models.ChartSet.id).where(
            models.ChartSet.source == chart.source, models.ChartSet.version == chart.version
        )
    )
    if exists is not None:
        raise ValueError(f"chart set {chart.source!r} version {chart.version!r} is already loaded")

    row = models.ChartSet(
        model=chart.model,
        ante=f"{chart.ante_type}:{chart.ante_size_bb:g}",
        source=chart.source,
        version=chart.version,
        eligible_for_analysis=chart.eligible_for_analysis,
    )
    session.add(row)
    session.flush()

    for node in chart.nodes:
        node_row = models.Node(
            chart_set_id=row.id,
            key=node.key,
            # Временные бакеты: точные значения узла. Сетку задаст тикет 15.
            stack_bucket=f"{node.stack_bb:g}bb",
            sizing_bucket=",".join(f"{p}:{s:g}" for p, s in sorted(node.sizings.items())),
            sizings=node.sizings,
        )
        session.add(node_row)
        session.flush()
        session.add_all(
            models.NodeStrategy(node_id=node_row.id, hand_class=hand, action=action, freq=freq)
            for hand, freqs in node.strategy.items()
            for action, freq in freqs.items()
        )
    return row
