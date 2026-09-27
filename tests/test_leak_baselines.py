"""T2-2 の B2/B3 前処理の単体検査。cwd=paper2/ で `pytest test_leak_baselines.py`。"""
import numpy as np
import pytest

from leakdec.decode.bposd_baseline import classification_error_from_llr
from leakdec.sim.iq_model import STATE_0, STATE_1, STATE_2, IQModel
from leakdec.decode.leak_baselines import (FlagStats, bayes_optimal_p_s, hanisch_outlier_flags, leak_posterior,
                            optimal_leak_flags, p_s_with_leak_flags)


@pytest.fixture(scope="module")
def model():
    return IQModel(snr=8.4, beta=0.05)


def test_hanisch_flags_far_points_not_means(model):
    iq = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 3.0], [4.0, 0.0]])
    flags = hanisch_outlier_flags(iq, model)
    assert flags.tolist() == [False, False, False, True, True]


def test_hanisch_boundary_is_circular_around_state0(model):
    # |0> 側の 1% 等高線：半径 sqrt(2 ln 100) σ。少し内側は非外れ値、少し外側は外れ値
    r = np.sqrt(2 * np.log(100)) * model.sigma
    for angle in (np.pi / 4, np.pi / 2):  # |1> の円に入らない方向
        inside = np.array([1.0 + 0.95 * r * np.cos(angle), 0.95 * r * np.sin(angle)])
        outside = np.array([1.0 + 1.05 * r * np.cos(angle), 1.05 * r * np.sin(angle)])
        assert not hanisch_outlier_flags(inside, model)
        assert hanisch_outlier_flags(outside, model)


def test_optimal_flags_use_prior(model):
    origin = np.array([[0.0, 0.0]])
    assert optimal_leak_flags(origin, model, leak_prior=0.1)[0]
    assert not optimal_leak_flags(origin, model, leak_prior=1e-6)[0]
    assert not optimal_leak_flags(np.array([[1.0, 0.0]]), model, leak_prior=0.1)[0]


def test_optimal_flags_match_posterior_argmax(model):
    rng = np.random.default_rng(0)
    iq = rng.normal(size=(5000, 2)) * 1.5
    post2 = leak_posterior(iq, model, 0.1)
    p0 = np.exp(model.log_pdf(iq, STATE_0)) * 0.45
    p1 = np.exp(model.log_pdf(iq, STATE_1)) * 0.45
    p2 = np.exp(model.log_pdf(iq, STATE_2)) * 0.1
    np.testing.assert_allclose(post2, p2 / (p0 + p1 + p2), rtol=1e-8)
    flags = optimal_leak_flags(iq, model, 0.1)
    np.testing.assert_array_equal(flags, (p2 > p0) & (p2 > p1))


def test_optimal_detector_recall_exceeds_circular_on_model_samples(model):
    """Hanisch 自身の指摘：円形境界はリークを過小評価する。"""
    rng = np.random.default_rng(1)
    rho = 0.09
    n = 200_000
    state = np.where(rng.random(n) < rho, STATE_2, rng.integers(0, 2, n))
    iq = model.sample(state, rng)
    truth = state == STATE_2
    b2 = FlagStats.of(hanisch_outlier_flags(iq, model), truth)
    b3 = FlagStats.of(optimal_leak_flags(iq, model, rho), truth)
    # 中点・同分散のリークは 1 点からは原理的に判別しにくい（最適でも再現率 ~0.17）。
    # それでも円形境界（~0.007）より桁違いに高い
    assert b3.recall > 10 * b2.recall
    assert b3.precision > 3 * b2.precision
    assert 0.1 < b3.recall < 0.3


def test_bayes_optimal_p_s_is_half_for_certain_leak_and_matches_llr_without_leak(model):
    rho = 0.09
    # リーク確実：中点で leak_std を極小にした点 → p_S → 0.5
    tiny = IQModel(snr=8.4, beta=0.05, leak_std=1e-3)
    p = bayes_optimal_p_s(np.array([[0.0, 0.0]]), tiny, rho)
    assert abs(p[0] - 0.5) < 1e-6
    # リーク事前確率 → 0 で 2 状態の p_S に一致
    rng = np.random.default_rng(2)
    iq = rng.normal(size=(1000, 2)) * 1.5
    p = bayes_optimal_p_s(iq, model, 1e-12)
    np.testing.assert_allclose(p, classification_error_from_llr(model.llr(iq)), rtol=1e-6, atol=1e-9)


def test_p_s_with_flags(model):
    llr = np.array([6.0, -6.0, 0.5])
    flags = np.array([False, True, False])
    p = p_s_with_leak_flags(llr, flags)
    assert p[1] == 0.5
    assert p[0] == pytest.approx(1 / (1 + np.exp(6.0)))
    assert p[2] == pytest.approx(1 / (1 + np.exp(0.5)))


def test_flag_stats():
    flags = np.array([True, True, False, False])
    truth = np.array([True, False, True, False])
    s = FlagStats.of(flags, truth)
    assert s.precision == 0.5 and s.recall == 0.5 and s.flagged_rate == 0.5
