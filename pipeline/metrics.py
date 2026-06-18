"""Scoring + inference helpers shared by the backtest and the models.

Probabilistic metrics (Brier, log-loss, AUC, PR-AUC), calibration (reliability
curve + ECE), serial-dependence-respecting CIs (stationary block bootstrap), and
predictive-ability tests (Diebold-Mariano, Giacomini-White).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

EPS = 1e-12


def _clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def brier(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p) -> float:
    y = np.asarray(y, float)
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p) -> float:
    """Mann-Whitney AUC; returns nan if only one class present."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pr_auc(y, p) -> float:
    """Average precision (area under precision-recall), trapezoid on recall."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


def reliability(y, p, n_bins: int = 10):
    """Return (bin_centers, observed_freq, mean_pred, counts) for a reliability curve."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, obs, pred, cnt = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        centers.append((lo + hi) / 2)
        if m.sum() == 0:
            obs.append(None); pred.append(None); cnt.append(0)
        else:
            obs.append(float(y[m].mean()))
            pred.append(float(p[m].mean()))
            cnt.append(int(m.sum()))
    return centers, obs, pred, cnt


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error (weighted |obs - pred| over bins)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    n = len(y)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            total += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(total)


def all_metrics(y, p) -> dict:
    return {
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "auc": auc(y, p),
        "pr_auc": pr_auc(y, p),
        "ece": ece(y, p),
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
    }


# --- stationary block bootstrap -------------------------------------------
def stationary_bootstrap_indices(n: int, mean_block: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap (geometric block lengths), vectorised.

    A new block starts at t=0 and with prob p=1/mean_block thereafter; within a
    block the index advances by one (wrapping mod n) from a random start.
    """
    p = 1.0 / max(mean_block, 1)
    restarts = rng.random(n) < p
    restarts[0] = True
    starts = rng.integers(0, n, size=n)
    block_start_pos = np.flatnonzero(restarts)        # positions where a block begins
    block_id = np.cumsum(restarts) - 1                # block index per position
    pos = np.arange(n)
    start_pos = block_start_pos[block_id]             # this position's block start
    offset = pos - start_pos
    base = starts[start_pos]                          # random start drawn at block start
    return (base + offset) % n


def bootstrap_ci(y, p, stat_fn, n_boot: int, block_len: int, seed: int,
                 alpha: float = 0.05):
    """Block-bootstrap CI for a metric of (y, p)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(n, block_len, rng)
        v = stat_fn(y[idx], p[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    return {
        "point": float(stat_fn(y, p)),
        "lo": float(np.quantile(vals, alpha / 2)),
        "hi": float(np.quantile(vals, 1 - alpha / 2)),
    }


# --- predictive-ability tests ----------------------------------------------
def _loss(y, p, kind="brier"):
    y = np.asarray(y, float)
    if kind == "brier":
        return (np.asarray(p, float) - y) ** 2
    pc = _clip(p)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def diebold_mariano(y, p_model, p_bench, kind="brier", h: int = 1) -> dict:
    """DM test on the per-period loss differential (model - bench).

    Negative mean diff => model has lower loss (better). HAC (Newey-West) variance
    with bandwidth h-1 to respect overlap/serial dependence.
    """
    d = _loss(y, p_model, kind) - _loss(y, p_bench, kind)
    n = len(d)
    dbar = d.mean()
    # Newey-West long-run variance
    gamma0 = np.mean((d - dbar) ** 2)
    lrv = gamma0
    for lag in range(1, max(h, 1)):
        w = 1 - lag / max(h, 1)
        cov = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        lrv += 2 * w * cov
    se = np.sqrt(max(lrv, EPS) / n)
    dm = dbar / se
    pval = 2 * (1 - stats.norm.cdf(abs(dm)))
    return {"mean_loss_diff": float(dbar), "dm_stat": float(dm),
            "p_value": float(pval), "favors": "model" if dbar < 0 else "benchmark"}


def giacomini_white(y, p_model, p_bench, kind="brier") -> dict:
    """Unconditional GW predictive-ability test (Wald form on the loss diff).

    With a constant instrument this reduces to a chi-square(1) on the mean loss
    differential — a robust complement to DM for the equal-predictive-ability null.
    """
    d = _loss(y, p_model, kind) - _loss(y, p_bench, kind)
    n = len(d)
    dbar = d.mean()
    var = np.mean((d - dbar) ** 2)
    stat = n * dbar ** 2 / max(var, EPS)
    pval = 1 - stats.chi2.cdf(stat, df=1)
    return {"gw_stat": float(stat), "p_value": float(pval),
            "favors": "model" if dbar < 0 else "benchmark"}
