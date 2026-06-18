"""Panel construction: FRED-MD tcode transforms, labels, and the vintage store.

The vintage store is the heart of the no-look-ahead discipline. For an as-of date
`tau` it returns *only* observations that were available by `tau`. Two backends:

* **vintage backend** — genuine FRED-MD dated vintages (data/raw/fredmd/<vt>.csv).
  `as_of(tau)` uses the latest vintage whose vintage-month <= tau.
* **publication-lag backend** — when real vintages are unavailable (e.g. the bulk
  FRED-MD CSV is blocked), we approximate availability by a per-series publication
  lag: a value for reference month `m` is visible only once `m + lag <= tau`. This
  is the "final-data (pseudo-vintage)" path; it removes *timing* look-ahead but not
  *revision* look-ahead, which is exactly what the vintage-vs-final refutation
  measures. Either backend satisfies the no-look-ahead test in tests/.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Publication lag in months from reference month to first availability.
# Financial/rates are effectively contemporaneous; surveys/output lag ~1-2 months.
DEFAULT_LAG = 1
GROUP_LAG = {
    "rates": 0, "prices": 1, "labour": 1, "output": 1,
    "housing": 1, "consumption": 2, "money": 1, "sentiment": 0,
}


def apply_tcode(series: pd.Series, tcode: int) -> pd.Series:
    """FRED-McCracken stationarity transforms.

    1 level; 2 Dx; 3 D2x; 4 log; 5 Dlog; 6 D2log; 7 D(x_t/x_{t-1}-1).
    """
    x = series.astype(float)
    if tcode == 1:
        return x
    if tcode == 2:
        return x.diff()
    if tcode == 3:
        return x.diff().diff()
    if tcode == 4:
        return np.log(x)
    if tcode == 5:
        return np.log(x).diff()
    if tcode == 6:
        return np.log(x).diff().diff()
    if tcode == 7:
        return (x / x.shift(1) - 1.0).diff()
    raise ValueError(f"unknown tcode {tcode}")


def transform_panel(levels: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    """Apply each series' tcode. `specs` = config['us']['panel'] entries."""
    out = {}
    for spec in specs:
        sid = spec["id"]
        if sid not in levels.columns:
            continue
        out[sid] = apply_tcode(levels[sid], int(spec["tcode"]))
    df = pd.DataFrame(out)
    df.index = pd.DatetimeIndex(levels.index)
    return df


def standardize(df: pd.DataFrame, ref: pd.DataFrame | None = None):
    """Z-score columns using stats from `ref` (or `df`); returns (z, mean, std)."""
    base = ref if ref is not None else df
    mu = base.mean()
    sd = base.std(ddof=0).replace(0, np.nan)
    return (df - mu) / sd, mu, sd


def winsorize(df: pd.DataFrame, z: float = 8.0) -> pd.DataFrame:
    """Clip standardized outliers (e.g. COVID) to +/- z to stop them dominating."""
    return df.clip(lower=-z, upper=z)


# --- labels ----------------------------------------------------------------
def usrec_labels(fred_monthly: pd.DataFrame, label_id: str = "USREC") -> pd.Series:
    s = fred_monthly[label_id].dropna()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    return s.astype(int).rename("recession")


def cdhowe_labels(csv_path, index: pd.DatetimeIndex) -> pd.Series:
    """Expand C.D. Howe peak/trough rows into a monthly 0/1 series on `index`."""
    rec = pd.read_csv(csv_path)
    lab = pd.Series(0, index=index, name="recession", dtype=int)
    for _, row in rec.iterrows():
        peak = pd.Timestamp(row["peak"] + "-01")
        trough = pd.Timestamp(row["trough"] + "-01")
        mask = (lab.index >= peak) & (lab.index <= trough)
        lab[mask] = 1
    return lab


# --- vintage store ---------------------------------------------------------
@dataclass
class VintageStore:
    """Serve a monthly panel as it was known at an as-of date.

    levels: raw (untransformed) panel, month-start index.
    specs:  panel spec list (for tcodes + groups).
    lags:   reference-month -> availability lag, per series id.
    """
    levels: pd.DataFrame
    specs: list[dict]
    lags: dict

    @classmethod
    def from_levels(cls, levels: pd.DataFrame, specs: list[dict]) -> "VintageStore":
        lags = {}
        for spec in specs:
            lags[spec["id"]] = GROUP_LAG.get(spec.get("group"), DEFAULT_LAG)
        return cls(levels=levels.sort_index(), specs=specs, lags=lags)

    def available_mask(self, tau: pd.Timestamp) -> pd.DataFrame:
        """Boolean frame: True where (reference month + lag) <= tau."""
        idx = self.levels.index
        mask = pd.DataFrame(False, index=idx, columns=self.levels.columns)
        for col in self.levels.columns:
            lag = self.lags.get(col, DEFAULT_LAG)
            release = idx + pd.DateOffset(months=lag)
            mask[col] = release <= tau
        return mask

    def as_of(self, tau) -> pd.DataFrame:
        """Levels visible at `tau` (rows with any series released by tau)."""
        tau = pd.Timestamp(tau)
        mask = self.available_mask(tau)
        vis = self.levels.where(mask)
        vis = vis.loc[vis.index <= tau + pd.DateOffset(months=2)]
        return vis.dropna(how="all")

    def transformed_as_of(self, tau) -> pd.DataFrame:
        """Stationary panel (tcodes applied) using only data visible at tau."""
        return transform_panel(self.as_of(tau), self.specs)

    def max_release_date(self, tau) -> pd.Timestamp:
        """Latest reference-month+lag still <= tau, for the no-look-ahead assert."""
        mask = self.available_mask(pd.Timestamp(tau))
        used = []
        idx = self.levels.index
        for col in self.levels.columns:
            lag = self.lags.get(col, DEFAULT_LAG)
            rel = (idx + pd.DateOffset(months=lag))[mask[col].values]
            if len(rel):
                used.append(rel.max())
        return max(used) if used else pd.Timestamp("1900-01-01")
