"""リーク処理ベースライン B2 / B3：IQ 点ごとにリークを判定し、その測定を p_S = 0.5 にする。

B2（Hanisch arXiv:2411.16228 §II F）：計算状態 f^(0), f^(1) の**両方**で密度がピークの 1% 未満なら
外れ値 → リーク扱い。密度の等高線なので境界は円形（|1> は崩壊の裾で歪む）。
|2> の較正データを持たない実機での妥協であり、リークを過小評価する（同論文が自認）。

AUDIT(2026-09-16) T2-2 の受け入れ基準「B3 ≥ B2」は LER で判定して PASS したが、根拠（最適判別器が円形境界を
下回らない）は検出性能の話であり、検出で見ると B3 再現率 0.000 < B2 0.008 で逆転している（ρ=2.45% では MAP が
一度も |2> を選ばないため）。基準は空虚に通っているので TODO には「LER では同値、検出では退化」と書くこと。

B3（RULES R1-2、本当の対戦相手）：シミュレーションではリーク分布 f^(2) を我々が定義しているので、
事前確率 π = ((1-ρ)/2, (1-ρ)/2, ρ)（ρ = 定常リーク率）の下で 3 状態の事後確率を閉形式で計算し、
|2> が最大なら リーク扱い。同分散なら線形境界。**1 点だけを見る最適判別器**なので、
ラウンド間の持続性はどちらも使えない。学習型との差はそこに帰着する。

どちらも「リークと判定した測定の p_S を 0.5（LLR = 0）にし、それ以外は 2 状態 LLR」という
B1 の上に載る前処理。判定フラグは適合率・再現率の評価（T4-2）にも使う。
"""
from dataclasses import dataclass

import numpy as np

from leakdec.decode.bposd_baseline import classification_error_from_llr
from leakdec.sim.iq_model import STATE_0, STATE_1, STATE_2, IQModel


def hanisch_outlier_flags(iq: np.ndarray, model: IQModel, threshold: float = 0.01) -> np.ndarray:
    """B2：f^(0), f^(1) の両方でピーク比 threshold 未満なら True。iq (..., 2) → (...)。"""
    peaks = _peak_densities(model)
    log_thr0 = np.log(threshold * peaks[STATE_0])
    log_thr1 = np.log(threshold * peaks[STATE_1])
    return (model.log_pdf(iq, STATE_0) < log_thr0) & (model.log_pdf(iq, STATE_1) < log_thr1)


def _peak_densities(model: IQModel) -> dict[int, float]:
    """判別軸上 (I, Q=0) で密度の最大値を数値的に取る（|1> は崩壊でピークが下がる）。"""
    z = np.linspace(-1.0 - 6.0 * model.sigma, 1.0 + 6.0 * model.sigma, 20001)
    iq = np.stack([z, np.zeros_like(z)], axis=-1)
    return {s: float(np.exp(model.log_pdf(iq, s)).max()) for s in (STATE_0, STATE_1)}


def optimal_leak_flags(iq: np.ndarray, model: IQModel, leak_prior: float) -> np.ndarray:
    """B3：3 状態の事後確率で |2> が最大なら True。iq (..., 2) → (...)。"""
    if not (0.0 < leak_prior < 1.0):
        raise ValueError("0 < leak_prior < 1")
    log_prior = np.log([0.5 * (1 - leak_prior), 0.5 * (1 - leak_prior), leak_prior])
    scores = np.stack([model.log_pdf(iq, s) + log_prior[s] for s in (STATE_0, STATE_1, STATE_2)], axis=-1)
    return scores.argmax(axis=-1) == STATE_2


def three_state_posterior(iq: np.ndarray, model: IQModel, leak_prior: float) -> np.ndarray:
    """(P(0|iq), P(1|iq), P(2|iq)) を (..., 3) で返す。"""
    log_prior = np.log([0.5 * (1 - leak_prior), 0.5 * (1 - leak_prior), leak_prior])
    scores = np.stack([model.log_pdf(iq, s) + log_prior[s] for s in (STATE_0, STATE_1, STATE_2)], axis=-1)
    scores -= scores.max(axis=-1, keepdims=True)
    post = np.exp(scores)
    return post / post.sum(axis=-1, keepdims=True)


def leak_posterior(iq: np.ndarray, model: IQModel, leak_prior: float) -> np.ndarray:
    """P(|2> | iq)。学習型のリーク確率と比較するときの参照値。"""
    return three_state_posterior(iq, model, leak_prior)[..., STATE_2]


def bayes_optimal_p_s(iq: np.ndarray, model: IQModel, leak_prior: float) -> np.ndarray:
    """B3s：1 点から得られる Bayes 最適な p_S = P(硬判定が理想値と違う | iq)。

    硬判定 h は 2 状態 LLR の符号。P(誤り) = P(逆の計算状態 | iq) + ½ P(リーク | iq)
    （リーク中の測定は理想値と無相関なので、硬判定は確率 ½ で外れる）。
    B3（判定 → p_S = 0.5）より情報を捨てない、per-shot で最強の古典前処理。
    """
    post = three_state_posterior(iq, model, leak_prior)
    hard_is_one = model.llr(iq) < 0
    other = np.where(hard_is_one, post[..., STATE_0], post[..., STATE_1])
    return other + 0.5 * post[..., STATE_2]


def p_s_with_leak_flags(meas_llr: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """リーク判定した測定は p_S = 0.5、それ以外は 2 状態 LLR から。"""
    p_s = classification_error_from_llr(meas_llr)
    return np.where(flags, 0.5, p_s)


@dataclass(frozen=True)
class FlagStats:
    precision: float
    recall: float
    flagged_rate: float

    @staticmethod
    def of(flags: np.ndarray, truth: np.ndarray) -> "FlagStats":
        tp = int((flags & truth).sum())
        fp = int((flags & ~truth).sum())
        fn = int((~flags & truth).sum())
        return FlagStats(
            precision=tp / (tp + fp) if tp + fp else float("nan"),
            recall=tp / (tp + fn) if tp + fn else float("nan"),
            flagged_rate=float(flags.mean()),
        )
