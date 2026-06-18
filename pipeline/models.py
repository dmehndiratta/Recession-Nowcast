"""Estimators + feature builders for the recession nowcast.

All models share a common interface so the pseudo-real-time backtest can drive
them uniformly:

    m = SomeModel(...)
    m.fit(X_train, y_train)          # X: DataFrame months x features, y: 0/1 Series
    p = m.predict_proba(X_new)       # -> array of P(recession)

The ladder (transparent -> rich):
  * SpreadProbit  — Estrella-Mishkin term-spread probit (benchmark A)
  * SahmLogit     — real-time Sahm trigger mapped to a probability (benchmark B)
  * FactorProbit  — PCA-EM dynamic factors -> probit  (primary linear DFM)
  * MidasLogit    — elastic-net logistic on multi-lag indicators (penalised MIDAS)
  * GbmFactors    — gradient-boosted trees on factors (nonlinear comparator)

The full-sample mixed-frequency state-space DFM (statsmodels DynamicFactorMQ) is
fit separately in 03_analysis/dfm_statespace.py; here we use a fast PCA-EM factor
extraction so the expanding-window loop (hundreds of refits) stays tractable. Both
are documented in CLAUDE.md.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier


# --- factor extraction (PCA with EM for missing values) --------------------
def pca_em_factors(panel: pd.DataFrame, k: int, max_iter: int = 15):
    """Extract k static factors from a (months x series) panel with missing data.

    Standardize, iteratively impute missing entries with the rank-k SVD
    reconstruction (a simple EM), and return (factors_df, loadings, mean, std).
    This approximates the smoothed factors of a dynamic factor model and is the
    fast estimator used inside the backtest loop.
    """
    X = panel.copy()
    mu = X.mean()
    sd = X.std(ddof=0).replace(0, np.nan)
    Z = (X - mu) / sd
    mask = Z.notna().to_numpy()
    M = np.where(mask, Z.to_numpy(), 0.0)
    cols = Z.columns
    k = min(k, M.shape[1])
    for _ in range(max_iter):
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        recon = (U[:, :k] * S[:k]) @ Vt[:k]
        prev = M.copy()
        M = np.where(mask, M, recon)
        if np.nanmax(np.abs(M - prev)) < 1e-6:
            break
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    factors = U[:, :k] * S[:k]
    fdf = pd.DataFrame(factors, index=panel.index,
                       columns=[f"F{i+1}" for i in range(k)])
    evr = (S[:k] ** 2) / (S ** 2).sum()
    return fdf, Vt[:k], mu, sd, evr


# --- estimators ------------------------------------------------------------
class _LogitBase:
    """sklearn logistic workhorse (numerically stable in a long refit loop)."""

    def __init__(self, C: float = 1.0, penalty: str = "l2",
                 l1_ratio: float | None = None):
        # No class reweighting: for probabilistic forecasting we want CALIBRATED
        # probabilities (Brier/log-loss/ECE), and class_weight='balanced' inflates
        # them toward 0.5, wrecking calibration. The ~13% imbalance is mild.
        self.kwargs = dict(C=C, penalty=penalty, solver="liblinear",
                           max_iter=2000)
        if penalty == "elasticnet":
            self.kwargs.update(solver="saga", l1_ratio=l1_ratio or 0.5)
        self.clf = None
        self.const_p = None
        self.cols = None
        self.mu = None
        self.sd = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.dropna()
        y = y.reindex(X.index)
        keep = y.notna()
        X, y = X[keep], y[keep]
        self.cols = list(X.columns)
        if y.nunique() < 2 or len(y) < 30:
            self.const_p = float(np.clip(y.mean() if len(y) else 0.0, 1e-4, 1 - 1e-4))
            return self
        # Standardize so the penalty acts uniformly across features (key for the
        # high-dimensional MIDAS lag block) -> better-calibrated probabilities.
        self.mu = X.mean()
        self.sd = X.std(ddof=0).replace(0, 1.0)
        Z = ((X - self.mu) / self.sd).to_numpy()
        self.clf = LogisticRegression(**self.kwargs).fit(Z, y.to_numpy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.cols)
        Xf = X.ffill().fillna(0.0)
        if self.clf is None:
            return np.full(len(X), self.const_p if self.const_p is not None else 0.1)
        Z = ((Xf - self.mu) / self.sd).to_numpy()
        return self.clf.predict_proba(Z)[:, 1]


class SpreadProbit(_LogitBase):
    name = "spread_probit"


class SahmLogit(_LogitBase):
    name = "sahm"


class FactorProbit(_LogitBase):
    name = "dfm"


class MidasLogit(_LogitBase):
    """Penalised (L1/lasso) logistic on indicators at multiple lags.

    L1 (liblinear) gives the same sparse-selection behaviour as elastic-net but
    is ~30x faster to refit, which matters across hundreds of expanding-window
    re-estimations. The standalone full-sample diagnostic reports the selected
    (non-zero) features.
    """
    name = "midas"

    def __init__(self, C: float = 0.2):
        super().__init__(C=C, penalty="l1")


class GbmFactors:
    """Gradient-boosted trees on factors (+ spread/sahm) — nonlinear comparator."""
    name = "gbm"

    def __init__(self, seed: int = 0):
        self.model = XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            eval_metric="logloss", random_state=seed, n_jobs=2,
            verbosity=0)
        self.const_p = None
        self.cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.dropna()
        y = y.reindex(X.index)
        keep = y.notna()
        X, y = X[keep], y[keep]
        self.cols = list(X.columns)
        if y.nunique() < 2 or len(y) < 40:
            self.const_p = float(np.clip(y.mean() if len(y) else 0.0, 1e-4, 1 - 1e-4))
            return self
        # scale_pos_weight=1 (no reweighting) keeps probabilities calibrated.
        self.model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.cols)
        Xf = X.ffill().fillna(0.0).to_numpy()
        if self.const_p is not None:
            return np.full(len(X), self.const_p)
        return self.model.predict_proba(Xf)[:, 1]


# --- statsmodels probit for the faithful full-sample benchmark report ------
def statsmodels_probit(spread: pd.Series, y: pd.Series):
    """Estrella-Mishkin probit; returns (fitted_model, summary_dict) or None."""
    df = pd.concat([spread.rename("spread"), y.rename("y")], axis=1).dropna()
    if df["y"].nunique() < 2 or len(df) < 50:
        return None
    X = sm.add_constant(df[["spread"]])
    try:
        res = sm.Probit(df["y"], X).fit(disp=0, maxiter=200)
    except Exception:  # noqa: BLE001
        return None
    return res, {
        "const": float(res.params["const"]),
        "spread_coef": float(res.params["spread"]),
        "spread_pvalue": float(res.pvalues["spread"]),
        "pseudo_r2": float(res.prsquared),
        "n": int(len(df)),
    }
