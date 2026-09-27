"""B4：ラウンド方向の HMM でリーク事後確率 P(ℓ_r | μ_1…μ_T) を閉形式で出す古典ベースライン。

補助量子ビットごとに 2 状態（正常 / リーク）の隠れマルコフ連鎖。遷移は LeakageModel と同じ
    P(1|0) = p_leak,  P(1|1) = p_stay_leaked = 1 - p_seep(1 - p_leak)
初期分布は定常率（サンプラも init="stationary"）。放射確率は各ラウンドの IQ 点に対して
    e_r(正常) = ½ f^(0)(μ_r) + ½ f^(1)(μ_r),   e_r(リーク) = f^(2)(μ_r)
（計算状態の事前確率は B1 / B3s と同じ等確率。実際の補助ビットは X 補助 25% / Z 補助 50% が 1 で、
ラウンド間で相関もあるが、その簡略化は B1 / B3s と共通にして条件を揃える）。

前向き後向きで事後確率を出し、B3s と同じ式で p_S に変える：
    p_S,r = (1 - π_r) · P(逆の計算状態 | μ_r, 正常) + ½ π_r,   π_r = P(ℓ_r = 1 | μ_1…μ_T)
ℓ_r を条件づければ μ_r の計算状態は他ラウンドと独立なので、第 1 項は 1 点の 2 状態 LLR から出る。

これは「その量子ビットの IQ 系列だけ」から得られる Bayes 最適であり、全体の理論上限ではない。
計算状態のビットを等確率・ラウンド間独立と置いているので、(a) 補助ビットがラウンド間で相関する
（検出器の発火率 ≈12%）、(b) 72 量子ビットのシンドロームには符号のパリティ構造がある、という情報は
使っていない。学習型はこの 2 つを使えるので B4 を超えうる（実測：step 10000 の raw モデル AUC 0.969 対
B4 0.958）。B4 は「量子ビットごとの時間方向推論の上限」として位置づける。p_seep = 1 では B3s に厳密一致。
拡張案 B4'：状態を (計算ビット, リーク) の 4 状態にしてビットの遷移（検出器発火率）も入れると (a) を回収できる。
"""
import numpy as np
from scipy.special import logsumexp

from leakdec.decode.bposd_baseline import classification_error_from_llr
from leakdec.sim.circuit_map import CircuitMap
from leakdec.sim.iq_model import STATE_0, STATE_1, STATE_2, IQModel
from leakdec.sim.leakage_model import LeakageModel


def aux_grid_index(cmap: CircuitMap) -> np.ndarray:
    """idx[q, r] = 補助測定のローカル添字 j（(shots, A) ↔ (shots, Q, R) の並べ替えに使う）。"""
    aux = cmap.aux_meas_positions
    q, r = cmap.aux_index[aux], cmap.meas_round[aux]
    idx = np.full((cmap.num_aux_qubits, cmap.num_aux_rounds), -1, dtype=np.int64)
    idx[q, r] = np.arange(len(aux))
    if (idx < 0).any():
        raise ValueError("補助測定が (量子ビット, ラウンド) の格子を埋めていない")
    return idx


def log_emissions(iq: np.ndarray, model: IQModel) -> np.ndarray:
    """(..., 2) の IQ 点 → log e(正常), log e(リーク) を (..., 2) で返す。"""
    l0 = model.log_pdf(iq, STATE_0)
    l1 = model.log_pdf(iq, STATE_1)
    normal = np.logaddexp(l0, l1) + np.log(0.5)
    return np.stack([normal, model.log_pdf(iq, STATE_2)], axis=-1)


def forward_backward(log_e: np.ndarray, leak: LeakageModel) -> np.ndarray:
    """log_e (..., R, 2) → 事後確率 P(リーク | 全ラウンド) (..., R)。対数空間の前向き後向き。"""
    R = log_e.shape[-2]
    pi = leak.stationary_rate
    log_init = np.log([1.0 - pi, pi])
    # log A[i, j] = log P(j | i)
    log_A = np.log([[1.0 - leak.p_leak, leak.p_leak],
                    [1.0 - leak.p_stay_leaked, leak.p_stay_leaked]])
    alpha = np.empty_like(log_e)
    beta = np.empty_like(log_e)
    alpha[..., 0, :] = log_init + log_e[..., 0, :]
    for r in range(1, R):
        # alpha_r(j) = e_r(j) + logsum_i [alpha_{r-1}(i) + log A[i, j]]
        alpha[..., r, :] = log_e[..., r, :] + logsumexp(alpha[..., r - 1, :, None] + log_A, axis=-2)
    beta[..., R - 1, :] = 0.0
    for r in range(R - 2, -1, -1):
        # beta_r(i) = logsum_j [log A[i, j] + e_{r+1}(j) + beta_{r+1}(j)]
        beta[..., r, :] = logsumexp(log_A + (log_e[..., r + 1, :] + beta[..., r + 1, :])[..., None, :], axis=-1)
    log_post = alpha + beta
    log_post -= logsumexp(log_post, axis=-1, keepdims=True)
    return np.exp(log_post[..., 1])


def hmm_leak_posterior(iq: np.ndarray, model: IQModel, leak: LeakageModel, cmap: CircuitMap,
                       log_e: np.ndarray | None = None) -> np.ndarray:
    """iq (S, A, 2) → P(ℓ_j = 1 | 同一量子ビットの全ラウンド) を補助測定順 (S, A) で返す。

    log_e (S, A, 2) を渡すと放射確率をそれで置き換える（テストでオラクルを入れる用）。
    """
    idx = aux_grid_index(cmap)
    if log_e is None:
        log_e = log_emissions(iq, model)
    post_grid = forward_backward(log_e[:, idx], leak)  # (S, Q, R)
    out = np.empty(post_grid.shape[:1] + (idx.size,), dtype=np.float64)
    out[:, idx] = post_grid
    return out


def hmm_p_s(meas_llr: np.ndarray, leak_post: np.ndarray) -> np.ndarray:
    """B4 の p_S = (1 - π) · P(誤り | μ, 正常) + ½ π。evaluate.py の hybrid と同じ式。"""
    return (1.0 - leak_post) * classification_error_from_llr(meas_llr) + 0.5 * leak_post
