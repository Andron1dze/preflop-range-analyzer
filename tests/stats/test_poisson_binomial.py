import math

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from core.stats.engine import Observation, StatsConfig, analyze_node, bh_adjust, hand_group, poisson_binomial_test, two_sided_p, zone_of

# --- математика Пуассона–Бернулли --------------------------------------------


def test_weighted_expectation_variance_and_z_by_hand():
    # E = 0.2 + 0.5 + 2·0.9 = 2.5; Var = 0.16 + 0.25 + 4·0.09 = 0.77; факт = 1 + 0 + 2 = 3.
    result = poisson_binomial_test(p=[0.2, 0.5, 0.9], did=[True, False, True], w=[1, 1, 2], exact_below=0)
    assert result.n == 3
    assert result.expected == pytest.approx(2.5)
    assert result.variance == pytest.approx(0.77)
    assert result.observed == pytest.approx(3.0)
    assert result.z == pytest.approx(0.5 / math.sqrt(0.77))


def test_weights_default_to_one():
    result = poisson_binomial_test(p=[0.5, 0.5], did=[True, True], exact_below=0)
    assert (result.expected, result.variance, result.observed) == (1.0, 0.5, 2.0)


def test_exact_gto_frequency_gives_zero_z():
    # 100 решений с p = 0.3 и ровно 30 действиями.
    did = [True] * 30 + [False] * 70
    assert poisson_binomial_test(p=[0.3] * 100, did=did).z == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(5))
def test_playing_gto_gives_small_z(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, size=5000)
    did = rng.uniform(0, 1, size=5000) < p
    assert abs(poisson_binomial_test(p=p, did=did).z) < 3


def test_z_is_standard_normal_under_gto():
    rng = np.random.default_rng(42)
    zs = []
    for _ in range(400):
        p = rng.uniform(0.05, 0.95, size=200)
        zs.append(poisson_binomial_test(p=p, did=rng.uniform(0, 1, size=200) < p).z)
    assert abs(np.mean(zs)) < 0.2
    assert np.std(zs) == pytest.approx(1.0, abs=0.15)


def test_overfolding_exceeds_threshold():
    # GTO сбрасывает 40%, игрок — 60%.
    rng = np.random.default_rng(7)
    folds = rng.uniform(0, 1, size=500) < 0.6
    result = poisson_binomial_test(p=[0.4] * 500, did=folds)
    assert result.z > StatsConfig().z_threshold
    assert result.z == pytest.approx(100 / math.sqrt(120), abs=2.5)


def test_underfolding_gives_negative_z():
    did = [True] * 20 + [False] * 80
    assert poisson_binomial_test(p=[0.4] * 100, did=did).z < -StatsConfig().z_threshold


EPS = StatsConfig().p_epsilon


def test_default_epsilon():
    assert EPS == 1e-4


def test_pure_strategies_matching_expectation_give_zero():
    assert poisson_binomial_test(p=[0.0, 1.0], did=[False, True], exact_below=0).z == pytest.approx(0.0, abs=1e-9)


def test_pure_strategy_deviation_is_large_but_finite():
    # Частоты ограничены [ε, 1−ε]: действие с GTO-частотой 0 даёт конечный z.
    z = poisson_binomial_test(p=[0.0, 0.0], did=[True, False], exact_below=0).z
    assert math.isfinite(z)
    assert z == pytest.approx((1 - 2 * EPS) / math.sqrt(2 * EPS * (1 - EPS)))
    assert poisson_binomial_test(p=[1.0], did=[False], exact_below=0).z == pytest.approx(-math.sqrt((1 - EPS) / EPS))


def test_normal_approximation_overstates_single_misclick():
    # Нормальное приближение: z = (1 − 500ε) / √(500ε(1−ε)) ≈ 4.25 — ложная «утечка».
    z = poisson_binomial_test(p=[0.0] * 500, did=[True] + [False] * 499, exact_below=0).z
    assert z == pytest.approx((1 - 500 * EPS) / math.sqrt(500 * EPS * (1 - EPS)))


def test_epsilon_is_configurable():
    z = poisson_binomial_test(p=[0.0] * 500, did=[True] + [False] * 499, eps=0.01, exact_below=0).z
    assert z == pytest.approx((1 - 5) / math.sqrt(5 * 0.99))


def test_epsilon_zero_restores_exact_pure_strategies():
    assert poisson_binomial_test(p=[0.0], did=[True], eps=0, exact_below=0).z == math.inf


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        poisson_binomial_test(p=[0.5], did=[True, False])
    with pytest.raises(ValueError):
        poisson_binomial_test(p=[1.2], did=[True])
    with pytest.raises(ValueError):
        poisson_binomial_test(p=[0.5], did=[True], w=[-1])


def test_empty_sample():
    result = poisson_binomial_test(p=[], did=[])
    assert (result.n, result.z) == (0, 0.0)


@pytest.mark.parametrize("z, expected", [(0, 1.0), (1.959964, 0.05), (-1.959964, 0.05), (math.inf, 0.0)])
def test_two_sided_p(z, expected):
    assert two_sided_p(z) == pytest.approx(expected, abs=1e-6)


# --- Benjamini–Hochberg ------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_bh_matches_statsmodels(seed):
    rng = np.random.default_rng(seed)
    pvalues = np.concatenate([rng.uniform(0, 1, 40), rng.uniform(0, 0.01, 10), [0.0, 1.0, 0.02, 0.02]])
    rng.shuffle(pvalues)
    expected = multipletests(pvalues, method="fdr_bh")[1]
    assert bh_adjust(list(pvalues)) == pytest.approx(list(expected))


def test_bh_edge_cases():
    assert bh_adjust([]) == []
    assert bh_adjust([0.03]) == [0.03]


# --- зоны и группы рук -------------------------------------------------------


@pytest.mark.parametrize(
    "p, zone",
    [(0.0, "never"), (0.099, "never"), (0.1, "mix"), (0.5, "mix"), (0.9, "mix"), (0.901, "pure"), (1.0, "pure")],
)
def test_zone_boundaries(p, zone):
    assert zone_of(p, StatsConfig()) == zone


@pytest.mark.parametrize(
    "hand, group",
    [
        ("AA", "pair"), ("22", "pair"),
        ("AKs", "suited_broadway"), ("JTs", "suited_broadway"),
        ("A5s", "suited_ace"),
        ("98s", "suited_connector"), ("97s", "suited_connector"), ("54s", "suited_connector"),
        ("96s", "suited_other"), ("K8s", "suited_other"),
        ("KQo", "offsuit_broadway"),
        ("A9o", "offsuit_ace"),
        ("72o", "offsuit_other"), ("98o", "offsuit_other"),
    ],
)
def test_hand_groups(hand, group):
    assert hand_group(hand) == group


# --- анализ узла -------------------------------------------------------------


def obs(hand, p, did, w=1.0, stage_known=True):
    return Observation(hand_class=hand, p=p, did=did, w=w, stage_known=stage_known)


def test_analyze_node_reports_top_level_zones_and_groups():
    rows = analyze_node(
        [obs("AA", 1.0, True), obs("72o", 0.0, True), obs("98s", 0.5, False), obs("KQo", 0.5, True)],
        StatsConfig(min_sample=1),
    )
    top = next(r for r in rows if r.zone is None and r.hand_group is None)
    assert top.n == 4
    assert top.expected == pytest.approx(2.0)
    assert {r.zone for r in rows if r.zone} == {"pure", "never", "mix"}
    assert {r.hand_group for r in rows if r.hand_group} == {"pair", "offsuit_other", "suited_connector", "offsuit_broadway"}
    never = next(r for r in rows if r.zone == "never")
    # Сыграл руку, которую GTO не играет никогда: большой, но конечный z.
    assert math.isfinite(never.z) and never.z > StatsConfig().z_threshold


def test_leak_requires_threshold_and_min_sample():
    overfold = [obs("72o", 0.4, i < 60) for i in range(100)]
    assert next(r for r in analyze_node(overfold, StatsConfig(min_sample=50)) if r.zone is None and r.hand_group is None).is_leak
    small = overfold[:10]
    top = next(r for r in analyze_node(small, StatsConfig(min_sample=50)) if r.zone is None and r.hand_group is None)
    assert top.is_leak is False


def test_unknown_stage_share():
    rows = analyze_node([obs("AA", 1.0, True, stage_known=False), obs("KK", 1.0, True)], StatsConfig(min_sample=1))
    top = next(r for r in rows if r.zone is None and r.hand_group is None)
    assert top.unknown_stage_share == pytest.approx(0.5)


# --- точный хвост при малом ожидаемом числе событий --------------------------


def binom_two_sided(k, n, p):
    from scipy.stats import binom

    return min(1.0, 2 * min(binom.sf(k - 1, n, p), binom.cdf(k, n, p)))


def test_default_exact_threshold():
    assert StatsConfig().exact_below == 5


def test_single_misclick_is_not_a_leak():
    result = poisson_binomial_test(p=[0.0] * 500, did=[True] + [False] * 499)
    assert result.exact
    assert two_sided_p(result.z) == pytest.approx(binom_two_sided(1, 500, EPS), rel=1e-6)
    assert abs(result.z) < StatsConfig().z_threshold


def test_repeated_misclicks_are_a_leak():
    result = poisson_binomial_test(p=[0.0] * 500, did=[True] * 5 + [False] * 495)
    assert result.exact
    assert result.z > StatsConfig().z_threshold


def test_missing_pure_action_once_is_not_a_leak():
    # Зеркальный случай: GTO всегда делает действие, игрок один раз не сделал — мало n − E.
    result = poisson_binomial_test(p=[1.0] * 500, did=[False] + [True] * 499)
    assert result.exact
    assert -StatsConfig().z_threshold < result.z < 0


@pytest.mark.parametrize("k", [0, 1, 2, 4, 8])
def test_exact_tail_matches_binomial_for_equal_p(k):
    result = poisson_binomial_test(p=[0.02] * 100, did=[True] * k + [False] * (100 - k))
    assert result.exact
    assert two_sided_p(result.z) == pytest.approx(binom_two_sided(k, 100, 0.02), rel=1e-6)
    assert math.copysign(1, result.z) == math.copysign(1, k - 2) or result.z == 0


def test_exact_tail_matches_enumeration_for_unequal_p():
    from itertools import product

    p = [0.01, 0.2, 0.05, 0.3, 0.02, 0.1]
    did = [True, False, True, False, False, False]
    k = sum(did)
    upper = lower = 0.0
    for outcome in product([0, 1], repeat=len(p)):
        prob = math.prod(pi if o else 1 - pi for pi, o in zip(p, outcome))
        upper += prob if sum(outcome) >= k else 0
        lower += prob if sum(outcome) <= k else 0
    expected = min(1.0, 2 * min(upper, lower))
    result = poisson_binomial_test(p=p, did=did)
    assert result.exact
    assert two_sided_p(result.z) == pytest.approx(expected, rel=1e-9)


def test_exact_regime_counts_without_weights():
    result = poisson_binomial_test(p=[0.0] * 100, did=[True] + [False] * 99, w=[3.0] * 100)
    assert result.exact
    assert (result.expected, result.observed) == (pytest.approx(100 * EPS), 1.0)


def test_large_expectation_uses_normal_approximation():
    result = poisson_binomial_test(p=[0.4] * 100, did=[True] * 40 + [False] * 60)
    assert not result.exact


def test_extreme_exact_tail_stays_finite():
    result = poisson_binomial_test(p=[0.0] * 400, did=[True] * 400)
    assert result.exact
    assert result.z > 30


def test_node_rows_report_exact_method():
    rows = analyze_node([obs("72o", 0.0, i == 0) for i in range(500)], StatsConfig(min_sample=1))
    top = next(r for r in rows if r.zone is None and r.hand_group is None)
    assert top.exact and not top.is_leak
