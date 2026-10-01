from pathlib import Path

import pytest

from core.charts.format import load_chart_file

CHARTS = sorted((Path(__file__).parents[2] / "charts").glob("*.y*ml"))


@pytest.mark.parametrize("path", CHARTS, ids=[p.name for p in CHARTS])
def test_repository_charts_are_valid(path):
    load_chart_file(path)


def test_approximate_core_chart_is_not_eligible_for_analysis():
    # Частоты — приближение по памяти, а не выход HRC.
    chart = load_chart_file(Path(__file__).parents[2] / "charts" / "40bb_core.yaml")
    assert chart.eligible_for_analysis is False
    assert {n.key for n in chart.nodes} == {"40bb|HERO=UTG", "40bb|HERO=BTN", "40bb|BTN:open|HERO=BB"}
    assert (chart.ante_type, chart.ante_size_bb) == ("each", 0.125)
