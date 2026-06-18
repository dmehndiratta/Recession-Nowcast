"""Determinism: the synthetic panel and the factor extraction are seed-stable."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from models import pca_em_factors

ROOT = Path(__file__).resolve().parents[1]


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "demo", ROOT / "pipeline" / "01_fetch" / "make_demo_panel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_panel_is_deterministic():
    demo = _load_demo()
    a, _ = demo.build(20260615)
    b, _ = demo.build(20260615)
    pd.testing.assert_frame_equal(a, b)


def test_demo_panel_changes_with_seed():
    demo = _load_demo()
    a, _ = demo.build(1)
    b, _ = demo.build(2)
    assert not a.equals(b)


def test_pca_em_deterministic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 8)),
                     index=pd.date_range("2000-01", periods=200, freq="MS"))
    f1, *_ = pca_em_factors(X, 3)
    f2, *_ = pca_em_factors(X, 3)
    pd.testing.assert_frame_equal(f1, f2)
