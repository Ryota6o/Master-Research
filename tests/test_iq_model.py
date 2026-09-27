"""T1-2 受け入れ基準。cwd=paper2/ で `pytest test_iq_model.py`。"""
"「量子ビットを測定したときに得られるIQ信号を、状態ごとにどんな確率分布で説明できるか」をモデル化した"
import numpy as np
import pytest
from soft_information_models import SuperconductingPDF

from leakdec.sim.iq_model import STATE_0, STATE_1, STATE_2, IQModel


@pytest.fixture(scope="module")
def model():
    return IQModel(snr=10.0, beta=0.1) # 標準的なIQModelインスタンス


# ---- 受け入れ基準 1：β=0 で LLR = SNR·z に厳密一致 ------------------------------------

# 実装のLLRが理論値SNRと機械精度レベルで一致するのかを見ている
def test_llr_equals_snr_times_i_when_beta_zero():
    m = IQModel(snr=7.3, beta=0.0)
    rng = np.random.default_rng(0)
    iq = rng.normal(size=(1000, 2)) * 3
    np.testing.assert_allclose(m.llr(iq), 7.3 * iq[:, 0], rtol=0, atol=1e-12)

# LLRはI軸のみに依存することを確認するテスト 
def test_llr_depends_only_on_i_axis(model):
    rng = np.random.default_rng(1)
    i = rng.normal(size=500)
    iq_a = np.stack([i, rng.normal(size=500)], axis=-1)
    iq_b = np.stack([i, rng.normal(size=500)], axis=-1)
    np.testing.assert_allclose(model.llr(iq_a), model.llr(iq_b), atol=1e-12)


# ---- Riverlane 式(7)(8) との一致（公式実装と直接比較） --------------------------------

# Softパッケージと自分で実装したIQModelの確率密度関数が一致するかを確認するテスト
@pytest.mark.parametrize("snr,beta", [(10.0, 0.0), (10.0, 0.1), (4.0, 0.5), (50.0, 0.05)])
def test_pdf_i_matches_riverlane_reference(snr, beta):
    ref = SuperconductingPDF(snr=snr, beta=beta)
    m = IQModel(snr=snr, beta=beta)
    z = np.linspace(-4, 4, 801)
    np.testing.assert_allclose(m.pdf_i(z, STATE_0), ref._0_state_pdf(z), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(m.pdf_i(z, STATE_1), ref._1_state_pdf(z), rtol=1e-8, atol=1e-14)

# 確率密度関数が全域で積分すると１になっていることを確認するテスト
@pytest.mark.parametrize("state", [STATE_0, STATE_1, STATE_2])
def test_pdf_i_normalised(model, state):
    z = np.linspace(-12, 12, 200001)
    assert abs(np.trapezoid(model.pdf_i(z, state), z) - 1.0) < 1e-6


def test_2d_pdf_projects_to_1d(model):
    q = np.linspace(-8, 8, 4001)
    for state in (STATE_0, STATE_1, STATE_2):
        for i in (-1.5, -0.3, 0.0, 0.7, 1.2):
            iq = np.stack([np.full_like(q, i), q], axis=-1)
            marginal = np.trapezoid(model.pdf(iq, state), q)
            assert abs(marginal - model.pdf_i(i, state)) < 1e-8


# ---- 受け入れ基準 2：サンプルのヒストグラムが解析式と一致（最大差 < 0.01） -------------
# モンテカルロサンプルの分布と、解析的に導いた確率密度関数が一致するかの確認。

# 200万サンプルで乱数を生成し、ヒストグラムが解析的ンア密度関数とどれだけ近いかを見ている
@pytest.mark.parametrize("state", [STATE_0, STATE_1, STATE_2])
def test_sample_histogram_matches_density(state):
    m = IQModel(snr=10.0, beta=0.3, leak_mean=(0.1, 0.4), leak_std=0.5)
    rng = np.random.default_rng(123)
    n = 2_000_000
    iq = m.sample(np.full(n, state), rng)
    edges = np.arange(-4.0, 4.0 + 1e-9, 0.05)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_i, _ = np.histogram(iq[:, 0], bins=edges, density=True)
    assert np.max(np.abs(hist_i - m.pdf_i(centers, state))) < 0.01
    # Q 周辺分布：計算状態は N(0,σ²)、リークは N(leak_mean[1], leak_std²)
    hist_q, _ = np.histogram(iq[:, 1], bins=edges, density=True)
    mu_q, sg = ((m.leak_mean[1], m.leak_sigma) if state == STATE_2 else (0.0, m.sigma))
    ref_q = np.exp(-0.5 * ((centers - mu_q) / sg) ** 2) / np.sqrt(2 * np.pi * sg**2)
    assert np.max(np.abs(hist_q - ref_q)) < 0.01


def test_sample_beta_zero_state1_is_plain_gaussian():
    m = IQModel(snr=10.0, beta=0.0)
    rng = np.random.default_rng(5)
    iq = m.sample(np.ones(500_000, dtype=int), rng)
    assert abs(iq[:, 0].mean() + 1.0) < 3e-3
    assert abs(iq[:, 0].std() - m.sigma) < 3e-3


# ---- 受け入れ基準 3：読み出し誤り率が τ_M に対して極小を持つ --------------------------

# 測定時間 τ_M を変化させたときに、読み出し誤り率が極小を持つことを確認する。
# 測定時間 τ_M が短すぎると SNR が小さくなり、長すぎると T1 減衰で誤り率が増えるため。
def test_readout_error_has_interior_minimum_in_measurement_time():
    t_f, t1 = 100e-9, 30e-6
    t_ms = np.geomspace(50e-9, 20e-6, 40)
    errs = np.array([IQModel.from_times(t, t_f, t1).readout_error_probability() for t in t_ms])
    k = int(np.argmin(errs))
    assert 0 < k < len(t_ms) - 1
    assert errs[0] > errs[k] and errs[-1] > errs[k]

# 緩和なしのとき、標準的なガウス分布の閉形式解と一致することを確認する。
def test_readout_error_matches_gaussian_closed_form_when_beta_zero():
    from math import erfc, sqrt
    m = IQModel.from_times(500e-9, 100e-9, t1=np.inf)
    assert abs(m.snr - 10.0) < 1e-12 and m.beta == 0.0
    assert abs(m.readout_error_probability() - 0.5 * erfc(sqrt(10.0) / 2)) < 1e-6

# リファレンスのSoftパッケージの実装と読み出し誤り率が一致することを確認する。
def test_readout_error_matches_riverlane_reference():
    m = IQModel(snr=10.0, beta=0.1)
    ref = SuperconductingPDF(snr=10.0, beta=0.1, num_sampling_intervals=200_000)
    assert abs(m.readout_error_probability() - ref.readout_error_probability) < 2e-4


# ---- 数値安定性 ---------------------------------------------------------------------

# 論文の式(8),(9),(10)の計算で erfc(a1) - erfc(a2) の計算をしないといけない。
# ガウス分布の裾でアンダーフローしたり、近いerfc同士をの差を計算すると、桁落ちになる可能性がある。
# そのため、「対数空間での計算」「反射公式（erfc(-x) = 2 - erfc(x))」を使えば、NaNやinfが出ないことを確認するテスト。
@pytest.mark.parametrize("snr,beta", [(10.0, 0.1), (1000.0, 0.5), (0.5, 2.0)])
def test_log_pdf_finite_for_extreme_inputs(snr, beta):
    m = IQModel(snr=snr, beta=beta)
    z = np.array([-60.0, -5.0, -1.0, 0.0, 1.0, 5.0, 60.0])
    for state in (STATE_0, STATE_1, STATE_2):
        assert np.all(np.isfinite(m.log_pdf_i(z, state)))
    iq = np.stack([z, np.full_like(z, 40.0)], axis=-1)
    assert np.all(np.isfinite(m.llr(iq)))
