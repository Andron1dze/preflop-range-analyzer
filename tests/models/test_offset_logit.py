import numpy as np
import pytest

from core.models.offset_logit import fit_irls, fit_statsmodels, logit

BETA_TRUE = np.array([0.3, -0.8, 0.5, 0.2])  # свободный член, два бинарных признака, непрерывный


def synthetic(seed=0, n=20000, beta=BETA_TRUE):
    """Решения с GTO-смещением logit(p) и заданными систематическими отклонениями β."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        np.ones(n),
        rng.integers(0, 2, n),
        rng.integers(0, 2, n),
        rng.normal(0, 1, n),
    ]).astype(float)
    offset = logit(rng.uniform(0.05, 0.95, n))
    weights = rng.uniform(0.5, 1.5, n)
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-(offset + X @ beta)))).astype(float)
    return X, y, offset, weights


@pytest.fixture(scope="module")
def data():
    return synthetic()


@pytest.fixture(scope="module")
def reference(data):
    X, y, offset, weights = data
    return fit_statsmodels(X, y, offset, weights)


# --- эталон statsmodels ------------------------------------------------------


def test_statsmodels_recovers_true_beta(reference):
    assert np.all(np.abs(reference.beta - BETA_TRUE) < 4 * reference.se)
    assert np.allclose(reference.beta, BETA_TRUE, atol=0.1)


def test_confidence_interval_covers_true_beta(reference):
    assert np.all(reference.ci_low < BETA_TRUE)
    assert np.all(BETA_TRUE < reference.ci_high)
    assert np.allclose(reference.ci_high - reference.beta, 1.959964 * reference.se)


# --- своя реализация на numpy ------------------------------------------------


def test_irls_recovers_true_beta(data):
    fit = fit_irls(*data)
    assert fit.converged
    assert np.allclose(fit.beta, BETA_TRUE, atol=0.1)


def test_irls_matches_statsmodels_without_penalty(data, reference):
    fit = fit_irls(*data, l2=0.0)
    assert np.allclose(fit.beta, reference.beta, rtol=1e-6, atol=1e-8)
    assert np.allclose(fit.se, reference.se, rtol=1e-6)
    assert np.allclose(fit.ci_low, reference.ci_low, rtol=1e-6, atol=1e-8)
    assert np.allclose(fit.ci_high, reference.ci_high, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("l2", [1e-8, 1e-4])
def test_tiny_penalty_barely_moves_coefficients(data, reference, l2):
    fit = fit_irls(*data, l2=l2, unpenalized=[0])
    assert np.max(np.abs(fit.beta - reference.beta)) < 10 * l2 + 1e-8


def test_penalty_shrinks_coefficients_but_not_intercept():
    X, y, offset, weights = synthetic(n=500)
    norms = []
    for l2 in (0.0, 10.0, 100.0, 1000.0):
        fit = fit_irls(X, y, offset, weights, l2=l2, unpenalized=[0])
        norms.append(np.linalg.norm(fit.beta[1:]))
    assert norms == sorted(norms, reverse=True)
    huge = fit_irls(X, y, offset, weights, l2=1e9, unpenalized=[0])
    assert np.allclose(huge.beta[1:], 0, atol=1e-5)
    assert abs(huge.beta[0]) > 0.05  # свободный член не штрафуется


def test_weights_default_to_one(data):
    X, y, offset, _ = data
    assert np.allclose(fit_irls(X, y, offset).beta, fit_statsmodels(X, y, offset).beta, rtol=1e-6)


def test_gto_player_betas_are_calibrated():
    # Игрок играет ровно по GTO: β/se ~ N(0, 1), 95%-интервалы накрывают 0 в ~95% случаев.
    zs, covered = [], []
    for seed in range(200):
        fit = fit_irls(*synthetic(seed=1000 + seed, n=2000, beta=np.zeros(4)))
        zs.extend(fit.beta / fit.se)
        covered.extend((fit.ci_low < 0) & (0 < fit.ci_high))
    assert abs(np.mean(zs)) < 0.1
    assert np.std(zs) == pytest.approx(1.0, abs=0.08)
    assert np.mean(covered) == pytest.approx(0.95, abs=0.02)


def test_offset_is_used():
    # Игрок играет ровно по GTO, но GTO-частота зависит от признака (например, позиции).
    # Без смещения модель приписала бы эту зависимость игроку как «отклонение».
    rng = np.random.default_rng(4)
    n = 20000
    X = np.column_stack([np.ones(n), rng.integers(0, 2, n)]).astype(float)
    offset = -1 + 2 * X[:, 1] + rng.normal(0, 0.5, n)
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-offset))).astype(float)

    with_offset = fit_irls(X, y, offset)
    no_offset = fit_irls(X, y, np.zeros(n))
    assert np.abs(with_offset.beta).max() < 0.1
    assert no_offset.beta[1] > 1.5


def test_inputs_are_validated(data):
    X, y, offset, weights = data
    with pytest.raises(ValueError):
        fit_irls(X, y[:-1], offset, weights)
    with pytest.raises(ValueError):
        fit_irls(X, y, offset, weights, l2=-1)
    with pytest.raises(ValueError):
        fit_irls(X, y * 2, offset, weights)


def test_logit():
    assert logit(np.array([0.5]))[0] == pytest.approx(0.0)
    assert logit(np.array([0.9]))[0] == pytest.approx(np.log(9))


def test_penalized_standard_errors_match_empirical_spread():
    # При L2 > 0 SE — сэндвич-оценка; она должна совпадать с реальным разбросом оценки.
    betas, ses = [], []
    for seed in range(300):
        fit = fit_irls(*synthetic(seed=5000 + seed, n=500), l2=50.0, unpenalized=[0])
        betas.append(fit.beta)
        ses.append(fit.se)
    empirical = np.std(betas, axis=0)[1:]
    reported = np.mean(ses, axis=0)[1:]
    # Сэндвич для штрафованной модели приближённый (~10%); формула без него ошибается на ~70%.
    assert reported == pytest.approx(empirical, rel=0.2)
