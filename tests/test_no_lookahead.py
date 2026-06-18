"""The cardinal test: the vintage store never reveals data released after tau."""
import numpy as np
import pandas as pd
import pytest

from common import load_config
from panel import VintageStore, transform_panel


@pytest.fixture(scope="module")
def store():
    cfg = load_config()
    idx = pd.date_range("1990-01", "2024-12", freq="MS")
    rng = np.random.default_rng(0)
    specs = cfg["us"]["panel"]
    data = {d["id"]: 100 + np.cumsum(rng.standard_normal(len(idx)))
            for d in specs}
    levels = pd.DataFrame(data, index=idx)
    return VintageStore.from_levels(levels, specs), cfg


def test_max_release_not_after_tau(store):
    vs, _ = store
    for tau in pd.date_range("2000-01", "2024-06", freq="12MS"):
        assert vs.max_release_date(tau) <= tau, f"look-ahead at {tau}"


def test_as_of_excludes_future_reference_months(store):
    vs, _ = store
    tau = pd.Timestamp("2010-06-01")
    asof = vs.as_of(tau)
    # every visible (reference month + lag) must be <= tau
    for col in asof.columns:
        lag = vs.lags.get(col, 1)
        visible = asof[col].dropna().index
        latest_release = (visible + pd.DateOffset(months=lag)).max()
        assert latest_release <= tau


def test_publication_lag_hides_recent_month(store):
    vs, _ = store
    tau = pd.Timestamp("2015-03-01")
    asof = vs.as_of(tau)
    # a series with lag>=1 must not show its tau-month value at tau
    lagged = [c for c in vs.levels.columns if vs.lags.get(c, 1) >= 1]
    assert lagged, "expected some lagged series"
    for col in lagged[:5]:
        assert pd.isna(asof.get(col, pd.Series(dtype=float)).get(tau, np.nan))


def test_transformed_as_of_is_subset_in_time(store):
    vs, _ = store
    tau = pd.Timestamp("2008-09-01")
    t = vs.transformed_as_of(tau)
    assert t.index.max() <= tau + pd.DateOffset(months=2)
