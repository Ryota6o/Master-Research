"""T1-4 受け入れ基準。cwd=paper2/ で `pytest test_boxplus.py`。"""
import numpy as np
import pytest

from leakdec.sim.boxplus import boxplus, detector_llrs
from leakdec.sim.circuit_map import CircuitMap, default_map_path


def _reference(a, b):
    return 2.0 * np.arctanh(np.tanh(a / 2.0) * np.tanh(b / 2.0))


def test_matches_reference_formula_for_moderate_values():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=10_000) * 5, rng.normal(size=10_000) * 5
    np.testing.assert_allclose(boxplus(a, b), _reference(a, b), rtol=1e-9, atol=1e-12)


def test_commutative():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=10_000) * 30, rng.normal(size=10_000) * 30
    np.testing.assert_array_equal(boxplus(a, b), boxplus(b, a))


def test_identity_and_annihilator():
    L = np.array([-70.0, -6.0, -0.5, 0.0, 0.5, 6.0, 70.0])
    np.testing.assert_allclose(boxplus(L, np.inf), L, atol=1e-12)
    np.testing.assert_allclose(boxplus(L, -np.inf), -L, atol=1e-12)
    np.testing.assert_allclose(boxplus(L, 0.0), 0.0, atol=1e-12)
    assert boxplus(np.inf, np.inf) == np.inf
    assert boxplus(np.inf, -np.inf) == -np.inf


def test_no_nan_or_inf_for_large_llr():
    L = np.array([50.0, -50.0, 100.0, 700.0, -1e6])
    out = boxplus(L[:, None], L[None, :])
    assert np.all(np.isfinite(out))
    assert out[0, 1] == pytest.approx(-50.0 + np.log(2.0), abs=1e-12)  # 50 ⊞ -50
    assert out[2, 3] == pytest.approx(100.0, abs=1e-12)  # 100 ⊞ 700


def test_reproduces_information_loss_table():
    """plan01 §3.5 の表。(6.0, 0.5) と (0.5, 6.0) が区別できない。"""
    assert boxplus(6.0, 0.5) == pytest.approx(0.50, abs=5e-3)
    assert boxplus(0.5, 6.0) == pytest.approx(0.50, abs=5e-3)
    assert boxplus(6.0, 6.0) == pytest.approx(5.31, abs=5e-3)
    assert boxplus(6.0, 3.0) == pytest.approx(2.95, abs=5e-3)
    assert boxplus(0.5, 0.5) == pytest.approx(0.12, abs=5e-3)
    assert boxplus(6.0, 0.0) == pytest.approx(0.00, abs=1e-12)


def test_detector_llrs_hard_decisions_agree_with_m2d():
    cmap = CircuitMap.load(default_map_path("bbc-72-12-6", "X", 6))
    rng = np.random.default_rng(2)
    aux_llr = rng.normal(size=(300, 576)) * 4
    det_llr = detector_llrs(aux_llr, cmap)
    assert det_llr.shape == (300, 504)
    hard_meas = np.zeros((300, 648), dtype=np.uint8)
    hard_meas[:, cmap.aux_meas_positions] = aux_llr < 0
    hard_det_expected = (hard_meas @ cmap.m2d.T.astype(np.uint8)) % 2
    np.testing.assert_array_equal((det_llr < 0).astype(np.uint8), hard_det_expected)
    with pytest.raises(ValueError):
        detector_llrs(rng.normal(size=(3, 648)), cmap)
