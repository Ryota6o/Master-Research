"""T1-5 受け入れ基準。cwd=paper2/ で `pytest test_soft_syndrome.py`。"""
import numpy as np
import pytest

from leakdec.sim.circuit_map import CircuitMap, default_map_path
from leakdec.sim.gym import QECGym
from leakdec.sim.iq_model import IQModel
from leakdec.sim.leakage_model import LeakageModel
from leakdec.sim.soft_syndrome import SoftSyndromeSampler


@pytest.fixture(scope="module")
def circuit():
    return QECGym("bbc-72-12-6", "X", "circuit", 0.004, num_rounds=6,
                  measure_both=True, load_saved_logical_ops=True)._circuit


@pytest.fixture(scope="module")
def cmap():
    return CircuitMap.load(default_map_path("bbc-72-12-6", "X", 6))


@pytest.fixture(scope="module")
def sampler(circuit, cmap):
    return SoftSyndromeSampler(circuit, IQModel(snr=10.0, beta=0.1),
                               LeakageModel(p_leak=0.02, p_seep=0.2), cmap)


def test_shapes(sampler):
    s = sampler.sample(shots=50, seed=0)
    assert s.ideal_meas.shape == (50, 648) and s.hard_meas.shape == (50, 648)
    assert s.detectors_ideal.shape == (50, 504)
    assert s.detectors_hard.shape == (50, 504)
    assert s.detectors_soft.shape == (50, 504)
    assert s.iq.shape == (50, 576, 2)
    assert s.meas_llr.shape == (50, 576)
    assert s.observables.shape == (50, 12)
    assert s.leaked.shape == (50, 8, 72) and s.leaked_aux.shape == (50, 576)


# ---- 受け入れ基準：固定シードで再現性 ---------------------------------------------------

def test_reproducible_with_fixed_seed(sampler):
    a, b = sampler.sample(200, seed=7), sampler.sample(200, seed=7)
    for name in ("ideal_meas", "hard_meas", "detectors_hard", "detectors_soft",
                 "iq", "meas_llr", "observables", "leaked", "leaked_aux"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def test_different_seeds_differ(sampler):
    a, b = sampler.sample(200, seed=7), sampler.sample(200, seed=8)
    assert not np.array_equal(a.ideal_meas, b.ideal_meas)
    assert not np.array_equal(a.iq, b.iq)
    assert not np.array_equal(a.leaked, b.leaked)


# ---- 受け入れ基準：3 表現が同一ショットから作られている --------------------------------

def test_three_representations_are_consistent_per_shot(sampler, cmap):
    s = sampler.sample(300, seed=1)
    # (iii) → LLR → (i)
    np.testing.assert_allclose(sampler.iq_model.llr(s.iq), s.meas_llr)
    np.testing.assert_array_equal(s.hard_meas[:, cmap.aux_meas_positions], s.meas_llr < 0)
    # (i) は hard_meas の M2D
    expect = (s.hard_meas.astype(np.uint8) @ cmap.m2d.T.astype(np.uint8)) % 2
    np.testing.assert_array_equal(s.detectors_hard, expect.astype(np.bool_))
    # (ii) の符号は (i) と一致
    np.testing.assert_array_equal(s.detectors_soft < 0, s.detectors_hard)
    # (ii) は boxplus(measurement pair)
    pairs = cmap.detector_aux_measurements()
    from leakdec.sim.boxplus import boxplus
    np.testing.assert_allclose(
        s.detectors_soft, boxplus(s.meas_llr[:, pairs[:, 0]], s.meas_llr[:, pairs[:, 1]]))


def test_data_readout_untouched_and_labels_from_ideal(sampler, cmap):
    s = sampler.sample(300, seed=2)
    data = ~cmap.is_aux
    np.testing.assert_array_equal(s.hard_meas[:, data], s.ideal_meas[:, data])
    det, obs = sampler._converter.convert(measurements=s.ideal_meas, separate_observables=True)
    np.testing.assert_array_equal(s.observables, obs)
    np.testing.assert_array_equal(s.detectors_ideal, det)


def test_leakage_statistics_and_iq_distribution(sampler):
    s = sampler.sample(2000, seed=3)
    assert abs(s.leaked_aux.mean() - sampler.leakage.stationary_rate) < 4e-3
    leaked_iq = s.iq[s.leaked_aux]
    assert len(leaked_iq) > 5000
    lm = sampler.iq_model.leak_mean
    assert abs(leaked_iq[:, 0].mean() - lm[0]) < 0.02
    assert abs(leaked_iq[:, 1].mean() - lm[1]) < 0.02
    # リークの硬判定 1 の割合 = ∫ f2 · 1[LLR<0] dz。β>0 では最尤境界が I>0 にずれるので 50% ではない
    from leakdec.sim.iq_model import STATE_2
    z = np.linspace(-6, 6, 120001)
    m = sampler.iq_model
    expect = np.trapezoid(m.pdf_i(z, STATE_2) * (m.llr(np.stack([z, 0 * z], -1)) < 0), z)
    assert expect > 0.52  # 実際にずれていることの確認
    got = s.hard_meas[:, sampler.cmap.aux_meas_positions][s.leaked_aux].mean()
    assert abs(got - expect) < 0.01
    # 非リークの補助測定は ideal 側に寄る（IQ 分類誤り率 ~ 数%）
    aux_ideal = s.ideal_meas[:, sampler.cmap.aux_meas_positions]
    aux_hard = s.hard_meas[:, sampler.cmap.aux_meas_positions]
    flip = (aux_ideal != aux_hard)[~s.leaked_aux].mean()
    assert 0.005 < flip < 0.05


def test_no_leakage_high_snr_reproduces_ideal_detectors(circuit, cmap):
    s = SoftSyndromeSampler(circuit, IQModel(snr=1e4, beta=0.0), None, cmap).sample(500, seed=4)
    np.testing.assert_array_equal(s.hard_meas, s.ideal_meas)
    np.testing.assert_array_equal(s.detectors_hard, s.detectors_ideal)
    assert not s.leaked.any()
