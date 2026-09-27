"""B4（HMM リーク事後確率）の単体検査。cwd=paper2/ で `pytest test_leak_hmm.py`。"""
import itertools

import numpy as np
import pytest

from leakdec.decode.bposd_baseline import classification_error_from_llr
from leakdec.sim.circuit_map import CircuitMap, default_map_path
from leakdec.learn.dataset import DataConfig
from leakdec.analysis.evaluate import auc
from leakdec.sim.iq_model import IQModel
from leakdec.decode.leak_baselines import bayes_optimal_p_s, leak_posterior, p_s_with_leak_flags
from leakdec.decode.leak_hmm import aux_grid_index, forward_backward, hmm_leak_posterior, hmm_p_s, log_emissions
from leakdec.sim.leakage_model import LeakageModel


@pytest.fixture(scope="module")
def cmap():
    return CircuitMap.load(default_map_path("bbc-72-12-6", "X", 6))


@pytest.fixture(scope="module")
def model():
    return IQModel(snr=8.4, beta=0.05)


def _brute_force_posterior(log_e, leak):
    """R ラウンドの全経路 2^R を列挙した事後確率（1 本の連鎖）。"""
    R = log_e.shape[0]
    pi = leak.stationary_rate
    A = np.array([[1 - leak.p_leak, leak.p_leak], [1 - leak.p_stay_leaked, leak.p_stay_leaked]])
    post = np.zeros(R)
    total = 0.0
    for path in itertools.product((0, 1), repeat=R):
        w = (pi if path[0] else 1 - pi) * np.exp(log_e[0, path[0]])
        for r in range(1, R):
            w *= A[path[r - 1], path[r]] * np.exp(log_e[r, path[r]])
        total += w
        post += w * np.array(path)
    return post / total


def test_forward_backward_matches_brute_force():
    rng = np.random.default_rng(0)
    leak = LeakageModel(p_leak=0.05, p_seep=0.3)
    log_e = rng.normal(size=(5, 6, 2))
    got = forward_backward(log_e, leak)
    for k in range(5):
        np.testing.assert_allclose(got[k], _brute_force_posterior(log_e[k], leak), rtol=1e-10, atol=1e-12)


def test_grid_index_covers_all_aux_measurements(cmap):
    idx = aux_grid_index(cmap)
    assert idx.shape == (72, 8)
    assert sorted(idx.ravel().tolist()) == list(range(576))


def test_p_seep_one_reduces_to_per_point_bayes_optimal(cmap, model):
    """持続長 1（i.i.d.）では HMM は 1 点の事後確率に厳密一致し、p_S は B3s に一致する。"""
    leak = LeakageModel(p_leak=0.0245, p_seep=1.0)
    s = DataConfig(p_leak=0.0245, p_seep=1.0).make_sampler().sample(50, seed=0)
    post = hmm_leak_posterior(s.iq, model, leak, cmap)
    np.testing.assert_allclose(post, leak_posterior(s.iq, model, leak.stationary_rate), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(hmm_p_s(s.meas_llr, post), bayes_optimal_p_s(s.iq, model, leak.stationary_rate),
                               rtol=1e-9, atol=1e-12)


def test_oracle_emissions_reproduce_orc(cmap, model):
    """放射確率に真の ℓ のデルタを入れると事後確率は ℓ 自身、p_S は ORC と一致する。"""
    leak = LeakageModel(p_leak=0.005, p_seep=0.2)
    s = DataConfig().make_sampler().sample(50, seed=1)
    log_e = np.where(s.leaked_aux[..., None], np.array([-1e3, 0.0]), np.array([0.0, -1e3]))
    post = hmm_leak_posterior(s.iq, model, leak, cmap, log_e=log_e)
    np.testing.assert_allclose(post, s.leaked_aux.astype(float), atol=1e-12)
    np.testing.assert_allclose(hmm_p_s(s.meas_llr, post), p_s_with_leak_flags(s.meas_llr, s.leaked_aux), atol=1e-12)


def test_hmm_beats_per_point_and_is_calibrated(cmap, model):
    """持続性ありでは HMM の AUC が 1 点判別を上回り、事後確率の平均が真のリーク率に一致する。"""
    leak = LeakageModel(p_leak=0.005, p_seep=0.2)
    s = DataConfig().make_sampler().sample(500, seed=2)
    post = hmm_leak_posterior(s.iq, model, leak, cmap)
    point = leak_posterior(s.iq, model, leak.stationary_rate)
    truth = s.leaked_aux
    a_hmm, a_point = auc(post.ravel(), truth.ravel()), auc(point.ravel(), truth.ravel())
    print(f"\n[B4] AUC hmm={a_hmm:.3f} per-point={a_point:.3f}  mean post={post.mean():.4f} truth={truth.mean():.4f}")
    assert a_hmm > a_point + 0.05
    assert abs(post.mean() - truth.mean()) < 0.003
    # 較正：事後確率で 10 分位に切り、各分位の真のリーク率が事後確率の平均に近い
    q = np.quantile(post.ravel(), [0.9, 0.99])
    hi = post.ravel() > q[1]
    assert abs(truth.ravel()[hi].mean() - post.ravel()[hi].mean()) < 0.05


def test_p_s_reduces_to_two_state_without_leak(model, cmap):
    s = DataConfig(p_leak=0.0).make_sampler().sample(20, seed=3)
    post = np.zeros_like(s.meas_llr)
    np.testing.assert_allclose(hmm_p_s(s.meas_llr, post), classification_error_from_llr(s.meas_llr))


def test_emission_shapes(model):
    iq = np.random.default_rng(4).normal(size=(3, 7, 2))
    e = log_emissions(iq, model)
    assert e.shape == (3, 7, 2) and np.all(np.isfinite(e))
