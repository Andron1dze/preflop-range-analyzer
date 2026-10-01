"""Логистическая регрессия со смещением (spec §9).

    logit P(действие) = logit(pᵢ) + β·x

GTO-частота входит фиксированным смещением, поэтому β читаются как систематические
отклонения игрока от солвера. Веса wᵢ (расстояние до узла) — веса наблюдений.

Две реализации: эталон на statsmodels и своя IRLS + L2 на numpy. При L2 = 0 они
совпадают, включая стандартные ошибки. При L2 > 0 ошибки — сэндвич-оценка
A⁻¹·I·A⁻¹, где I — информация Фишера, A = I + L2·diag(штрафуемые).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import statsmodels.api as sm
from sqlalchemy.orm import Session

from core.db.models import ModelCoef


@dataclass
class FitResult:
    beta: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    converged: bool
    iterations: int


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def fit_statsmodels(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    alpha: float = 0.05,
) -> FitResult:
    X, y, offset, weights = _validate(X, y, offset, weights)
    model = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset, var_weights=weights)
    result = model.fit(tol=1e-12, maxiter=200)
    ci = result.conf_int(alpha=alpha)
    return FitResult(
        beta=np.asarray(result.params),
        se=np.asarray(result.bse),
        ci_low=np.asarray(ci[:, 0]),
        ci_high=np.asarray(ci[:, 1]),
        converged=bool(result.converged),
        iterations=int(result.fit_history["iteration"]),
    )


def fit_irls(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    l2: float = 0.0,
    unpenalized: Sequence[int] = (),
    alpha: float = 0.05,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> FitResult:
    """IRLS с L2-штрафом; столбцы из `unpenalized` (обычно свободный член) не штрафуются."""
    X, y, offset, weights = _validate(X, y, offset, weights)
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    penalty = np.full(X.shape[1], l2)
    penalty[list(unpenalized)] = 0.0
    penalty = np.diag(penalty)

    beta = np.zeros(X.shape[1])
    converged = False
    for iteration in range(1, max_iter + 1):
        eta = offset + X @ beta
        mu = 1 / (1 + np.exp(-eta))
        variance = np.clip(mu * (1 - mu), 1e-12, None)
        working = X @ beta + (y - mu) / variance
        W = weights * variance
        new_beta = np.linalg.solve(X.T @ (W[:, None] * X) + penalty, X.T @ (W * working))
        step = np.max(np.abs(new_beta - beta))
        beta = new_beta
        if step < tol:
            converged = True
            break

    mu = 1 / (1 + np.exp(-(offset + X @ beta)))
    fisher = X.T @ ((weights * mu * (1 - mu))[:, None] * X)
    inverse = np.linalg.inv(fisher + penalty)
    covariance = inverse @ fisher @ inverse
    se = np.sqrt(np.diag(covariance))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return FitResult(beta, se, beta - z * se, beta + z * se, converged, iteration)


def save_coefs(
    session: Session, run_id: int, tree_step: str, features: Sequence[str], fit: FitResult
) -> None:
    if len(features) != len(fit.beta):
        raise ValueError(f"{len(features)} feature names for {len(fit.beta)} coefficients")
    session.add_all(
        ModelCoef(
            run_id=run_id,
            tree_step=tree_step,
            feature=feature,
            beta=float(fit.beta[i]),
            se=float(fit.se[i]),
            ci_low=float(fit.ci_low[i]),
            ci_high=float(fit.ci_high[i]),
        )
        for i, feature in enumerate(features)
    )


def _validate(X, y, offset, weights):
    X, y, offset = (np.asarray(a, dtype=float) for a in (X, y, offset))
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if X.ndim != 2 or not len(X) == len(y) == len(offset) == len(weights):
        raise ValueError("X, y, offset and weights must describe the same observations")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("y must be in [0, 1]")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    return X, y, offset, weights
