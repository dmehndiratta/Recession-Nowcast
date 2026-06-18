"""Labels join correctly: NBER/USREC and the C.D. Howe chronology expansion."""
import pandas as pd

from common import MANUAL
from panel import cdhowe_labels


def test_cdhowe_expands_peak_to_trough():
    idx = pd.date_range("1980-01", "2024-12", freq="MS")
    lab = cdhowe_labels(MANUAL / "cdhowe_recessions.csv", idx)
    # 2008-11 -> 2009-05 inclusive = 7 months
    window = lab.loc["2008-11":"2009-05"]
    assert window.sum() == 7 and window.min() == 1
    # COVID 2020-03 -> 2020-04 = 2 months
    assert lab.loc["2020-03":"2020-04"].sum() == 2
    # 2022 and 2023 were NOT recessions per the council
    assert lab.loc["2022-01":"2023-12"].sum() == 0


def test_cdhowe_is_binary_and_aligned():
    idx = pd.date_range("1980-01", "2024-12", freq="MS")
    lab = cdhowe_labels(MANUAL / "cdhowe_recessions.csv", idx)
    assert set(lab.unique()).issubset({0, 1})
    assert lab.index.equals(idx)
    assert lab.sum() > 0
