"""T1-6 健全性チェック（関門）。cwd=paper2/ で `pytest test_pipeline.py`。

3 番目の基準（soft 復号 B1 が hard 復号 B0 に勝つ）は `run_gate_t1_6.py` で実測し、
その結果ファイルをここで検査する。
"""
from pathlib import Path

import numpy as np
import pytest
from soft_information_models import SuperconductingPDF

from leakdec.sim.circuit_map import CircuitMap, default_map_path
from leakdec.sim.gym import QECGym
from leakdec.sim.iq_model import IQModel
from leakdec.sim.soft_syndrome import SoftSyndromeSampler

P = 0.004


@pytest.fixture(scope="module")
def gym72():
    return QECGym("bbc-72-12-6", "X", "circuit", P, num_rounds=6,
                  measure_both=True, load_saved_logical_ops=True)


@pytest.fixture(scope="module")
def cmap():
    return CircuitMap.load(default_map_path("bbc-72-12-6", "X", 6))


# ---- 基準 1：リーク無し・σ→0 で hard 検出器が既存パイプラインの検出器と一致 ---------------

def test_sigma_to_zero_reproduces_stim_detectors_exactly(gym72, cmap):
    s = SoftSyndromeSampler(gym72._circuit, IQModel(snr=1e6, beta=0.0), None, cmap).sample(2000, seed=0)
    np.testing.assert_array_equal(s.hard_meas, s.ideal_meas)
    np.testing.assert_array_equal(s.detectors_hard, s.detectors_ideal)
    np.testing.assert_array_equal(s.detectors_soft < 0, s.detectors_ideal)


def test_sigma_to_zero_matches_dem_sampler_statistics(gym72, cmap):
    """既存の DEM サンプラ（gym）と回路サンプラは別物なので分布で比較する。

    DEM は脱分極チャネルの各 Pauli 成分を独立な誤り機構として扱うため、回路サンプラより
    発火率がわずかに高い方向の系統差 O(p²) がある。実測では平均発火率 0.1209 対 0.1216（+0.6%）。
    """
    n = 20000
    s = SoftSyndromeSampler(gym72._circuit, IQModel(snr=1e6, beta=0.0), None, cmap).sample(n, seed=1)
    det_dem, obs_dem, _ = gym72.get_detector_error_model().compile_sampler(seed=11).sample(shots=n)
    r_ours, r_dem = s.detectors_hard.mean(axis=0), det_dem.mean(axis=0)
    se = np.sqrt(r_dem * (1 - r_dem) / n * 2) + 1e-9
    z = (r_ours - r_dem) / se
    assert np.abs(z).max() < 6.0, f"検出器発火率の最大 z={np.abs(z).max():.2f}"
    assert abs(r_ours.mean() - r_dem.mean()) / r_dem.mean() < 0.01
    f_ours, f_dem = s.observables.any(axis=1).mean(), obs_dem.any(axis=1).mean()
    assert abs(f_ours - f_dem) < 5 * np.sqrt(f_dem * (1 - f_dem) / n * 2)


# ---- 基準 2：積分後の分布が Riverlane 式(7)(8) と一致 -------------------------------------

def test_pipeline_iq_marginals_match_riverlane_pdf(gym72, cmap):
    snr, beta = 8.4, 0.05
    s = SoftSyndromeSampler(gym72._circuit, IQModel(snr=snr, beta=beta), None, cmap).sample(8000, seed=2)
    ref = SuperconductingPDF(snr=snr, beta=beta)
    ideal_aux = s.ideal_meas[:, cmap.aux_meas_positions]
    edges = np.arange(-4.0, 4.0 + 1e-9, 0.05)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for bit, pdf in ((0, ref._0_state_pdf), (1, ref._1_state_pdf)):
        i_vals = s.iq[ideal_aux == bit][:, 0]
        assert len(i_vals) > 1_500_000
        hist, _ = np.histogram(i_vals, bins=edges, density=True)
        assert np.max(np.abs(hist - pdf(centers))) < 0.01
    q_vals = s.iq[..., 1].ravel()
    sg = np.sqrt(2 / snr)
    hist_q, _ = np.histogram(q_vals, bins=edges, density=True)
    assert np.max(np.abs(hist_q - np.exp(-0.5 * (centers / sg) ** 2) / np.sqrt(2 * np.pi * sg**2))) < 0.01


# ---- 基準 3：soft 復号（B1）が hard 復号（B0）に勝つ（run_gate_t1_6.py の結果を検査） ---------

def test_soft_bposd_beats_hard_bposd():
    path = Path("results/gate_t1_6.npz")
    if not path.exists():
        pytest.skip("先に `python scripts/run_gate_t1_6.py` を実行する")
    r = np.load(path)
    n = int(r["shots"])
    fail_b0, fail_b1 = int(r["fail_b0"]), int(r["fail_b1"])
    assert fail_b1 < fail_b0, f"B1={fail_b1}/{n} は B0={fail_b0}/{n} に勝っていない"
    # 同一ショットでの対応のある比較：B0 だけ失敗 vs B1 だけ失敗（McNemar）
    only_b0, only_b1 = int(r["only_b0_fails"]), int(r["only_b1_fails"])
    assert only_b0 > only_b1 + 3 * np.sqrt(only_b0 + only_b1), "差が統計的に有意でない"
