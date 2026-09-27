"""3状態 (|0>, |1>, |2>) の IQ 平面測定モデル。

計算状態は Riverlane (arXiv:2504.03504) 式(7)(8) — ガウス + 測定窓内の振幅減衰の閉形式 —
を IQ 平面へ拡張したもの。判別軸 (I) 上に |0> = +1, |1> = -1 を置き、Q 成分は
状態によらない等方ガウス雑音。I への射影が 1 次元版と厳密に一致する。

リーク状態 |2> は計算状態の中間付近に置いた第3のガウス (Hanisch arXiv:2411.16228 §II F,
"a separate Gaussian distribution in the IQ plane, typically positioned between the
Gaussians representing the computational states")。位置と幅はパラメータ。

SNR の規約
----------
SNR = 2 τ_M / τ_F（Riverlane 公開コード `SuperconductingPDF` の docstring と同じ）。
（τ_M：測定時間, τ_F：揺らぎ時間, τ1：T1 減衰時間）
論文式(11) は τ_M / 2τ_F と読めるが、その規約だと τ_M=500ns, τ_F=100ns で読み出し誤り率が
約 13% と非現実的になる（コード規約なら約 1.3%）。plan01 §4.2 / RULES R5-3 に従いコード側を採用。
ガウスの分散は σ² = 2/SNR（|μ_0 - μ_1| = 2 に対する定義）。
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf, erfcx

STATE_0, STATE_1, STATE_2 = 0, 1, 2


def _log_erfc_pos(b: NDArray) -> NDArray:
    """b ≥ 0 に対する log(erfc(b))。erfcx 経由でアンダーフローしない。"""
    # erfcxとは・・・exp(x^2)算で、xが大きいときにerfc(x)がアンダーフローするのを防ぐための関数
    return np.log(erfcx(b)) - b**2


def _log_erfc_diff(a1: NDArray, a2: NDArray) -> NDArray:
    """a1 < a2 に対する log(erfc(a1) - erfc(a2))。

    両者が同符号で大きいと素朴な差は打ち消しで壊れる。反射 erfc(a) = 2 - erfc(-a) を使い、
    差を常に「正の引数の erfc の大小」として評価する。
    """
    a1 = np.asarray(a1, dtype=np.float64) # 配列
    a2 = np.asarray(a2, dtype=np.float64)
    out = np.empty(np.broadcast(a1, a2).shape, dtype=np.float64)

    # ブールマスクで「配列全体」に対して振り分けを行い、Trueの要素だけを計算する。forループで1つずつ計算するより高速。
    both_pos = a1 >= 0 # 801要素全てに対する真偽値の配列が得られる
    both_neg = a2 <= 0
    straddle = ~(both_pos | both_neg)

    "a1とa2がどのような関係になっているのかについて、3つのケースに分けて計算する。"
    # 0 ≤ a1 < a2: erfc(a1) - erfc(a2)、erfc(a1) が大きい側
    hi, lo = _log_erfc_pos(a1[both_pos]), _log_erfc_pos(a2[both_pos])
    out[both_pos] = hi + np.log1p(-np.exp(lo - hi))
    # a1 < a2 ≤ 0: erfc(-a2) - erfc(-a1)、0 ≤ -a2 < -a1 なので erfc(-a2) が大きい側
    hi, lo = _log_erfc_pos(-a2[both_neg]), _log_erfc_pos(-a1[both_neg])
    out[both_neg] = hi + np.log1p(-np.exp(lo - hi))
    # a1 < 0 < a2: erf(a2) - erf(a1)、原点付近の erf は打ち消しなしで正確
    out[straddle] = np.log(erf(a2[straddle]) - erf(a1[straddle]))
    return out


@dataclass(frozen=True)
class IQModel:
    snr: float
    beta: float
    leak_mean: tuple[float, float] = (0.0, 0.0)
    leak_std: float | None = None

    def __post_init__(self):
        if self.snr <= 0:
            raise ValueError("snr must be positive")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")

    @classmethod
    def from_times(cls, t_m: float, t_f: float, t1: float, **kw) -> "IQModel":
        """測定時間 τ_M、揺らぎ時間 τ_F、T1 から構成する。SNR = 2τ_M/τ_F, β = τ_M/T1。論文式(11) とは分母分子が逆。"""
        return cls(snr=2.0 * t_m / t_f, beta=t_m / t1, **kw)

    @property
    def sigma(self) -> float:
        return float(np.sqrt(2.0 / self.snr))

    @property
    def leak_sigma(self) -> float:
        return self.sigma if self.leak_std is None else float(self.leak_std)

    # ---- 1次元（判別軸 I 上）の密度 ------------------------------------------------

    # 対数計算にしないと、桁溢れとか桁落ちで NaN が出る可能性があるので、対数計算を基本にする。
    def log_pdf_i(self, z: NDArray, state: int) -> NDArray:
        """判別軸に射影した 1 次元密度の対数。state=0,1 は Riverlane 式(7)(8) に一致。"""
        z = np.asarray(z, dtype=np.float64) # 複数のIQ観測の配列
        s = self.snr
        log_amp = 0.5 * np.log(s / (4.0 * np.pi))
        # T1緩和なし。中心ｎが１、分散が2/snrのガウス分布。論文式(7)の第1項。
        if state == STATE_0:
            return log_amp - 0.25 * s * (z - 1.0) ** 2
        # T1緩和を考慮。
        if state == STATE_1:
            log_no_decay = log_amp - self.beta - 0.25 * s * (z + 1.0) ** 2
            if self.beta == 0.0: # 崩壊なしのときは論文式(8)の第2項は消えるので、崩壊なしのガウス分布を返す
                return log_no_decay
            # 式(8) の第2項: (β/4) exp(β²/4s + β(z-1)/2) [erfc(a1) - erfc(a2)]
            b = self.beta / (2.0 * np.sqrt(s))
            half_sqrt_s = 0.5 * np.sqrt(s)
            a1 = b + (z - 1.0) * half_sqrt_s
            a2 = b + (z + 1.0) * half_sqrt_s
            log_e = self.beta**2 / (4.0 * s) + 0.5 * self.beta * (z - 1.0)
            log_decay = np.log(0.25 * self.beta) + log_e + _log_erfc_diff(a1, a2)
            return np.logaddexp(log_no_decay, log_decay) # 「緩和しないケースの確率密度」「緩和するケースの確率密度」を足し算する。
        # リーク状態 |2> はガウス分布。論文式(10) の第1項。
        if state == STATE_2:
            sg = self.leak_sigma
            return -0.5 * np.log(2.0 * np.pi * sg**2) - 0.5 * ((z - self.leak_mean[0]) / sg) ** 2
        raise ValueError(f"unknown state {state}")

    def pdf_i(self, z: NDArray, state: int) -> NDArray:
        return np.exp(self.log_pdf_i(z, state))

    # ---- 2次元（IQ 平面）の密度 -----------------------------------------------------

    # 2次元の密度は、計算状態は Riverlane 式(7)(8) の拡張、リーク状態はガウス分布。
    def log_pdf(self, iq: NDArray, state: int) -> NDArray:
        """IQ 点 (..., 2) の密度の対数。Q 成分は状態によらず等方ガウス。"""
        iq = np.asarray(iq, dtype=np.float64)
        i, q = iq[..., 0], iq[..., 1]
        if state == STATE_2:
            sg = self.leak_sigma
            return (-np.log(2.0 * np.pi * sg**2)
                    - 0.5 * (((i - self.leak_mean[0]) ** 2 + (q - self.leak_mean[1]) ** 2) / sg**2))
        sg = self.sigma
        log_q = -0.5 * np.log(2.0 * np.pi * sg**2) - 0.5 * (q / sg) ** 2
        return self.log_pdf_i(i, state) + log_q

    def pdf(self, iq: NDArray, state: int) -> NDArray:
        return np.exp(self.log_pdf(iq, state))

    # ０である確からしさと１である確からしさの比を返す。正なら |0> 寄り、負なら |1> 寄り。
    def llr(self, iq: NDArray) -> NDArray:
        """2状態 LLR = log f0/f1。正なら |0> 寄り。β=0 で SNR·I に一致。"""
        iq = np.asarray(iq, dtype=np.float64)
        i = iq[..., 0]
        return self.log_pdf_i(i, STATE_0) - self.log_pdf_i(i, STATE_1)

    # ---- サンプリング -----------------------------------------------------------------
    # 乱数を使用して、実際にIQ観測点を生成する関数
    # state: 各shotがどの状態(0,1,2)であるかを表す配列
    def sample(self, state: NDArray, rng: np.random.Generator) -> NDArray:
        """状態 (…,) ∈ {0,1,2} ごとに IQ 点 (…, 2) を引く。"""
        state = np.asarray(state)
        shape = state.shape
        sg = self.sigma
        mean_i = np.empty(shape, dtype=np.float64) # I軸の平均値を格納する配列
        std = np.full(shape, sg, dtype=np.float64) # Q軸の標準偏差を格納する配列

        m0 = state == STATE_0
        m1 = state == STATE_1
        m2 = state == STATE_2
        if not np.all(m0 | m1 | m2):
            raise ValueError("state must be in {0,1,2}")

        mean_i[m0] = 1.0
        # |1>: 確率 e^{-β} で崩壊なし、それ以外は窓内の時刻 u∈[0,1) で崩壊し平均が 1-2u
        n1 = int(m1.sum()) # |1> の数
        if n1:
            decayed = rng.random(n1) < -np.expm1(-self.beta) # 崩壊するかどうかの判定 expm1は exp(x)-1 を計算する関数で、xが小さいときに桁落ちしないようにするための関数
            u = np.empty(n1) # uは崩壊する場合の平均値を計算するための変数
            u[~decayed] = 1.0  # 崩壊なし → 平均 -1
            if decayed.any():
                # 密度 β e^{-βu} / (1 - e^{-β}) on [0,1) の逆関数法
                # 一様乱数vを１個引くだけで、正しい確率分布に従う緩和時刻uが得られる。exp(-βu) = 1 + v (exp(-β) - 1) を解くと u = -log(1 + v (exp(-β) - 1)) / β
                v = rng.random(int(decayed.sum()))
                u[decayed] = -np.log1p(v * np.expm1(-self.beta)) / self.beta
            mean_i[m1] = 1.0 - 2.0 * u # 平均がこの間にある。
        mean_i[m2] = self.leak_mean[0]
        std[m2] = self.leak_sigma

        # 2というのは、I値とQ値の値を格納するための配列の次元数。
        out = np.empty(shape + (2,), dtype=np.float64)
        # 標準的なガウス分布からのサンプリングという意味になる
        out[..., 0] = mean_i + std * rng.standard_normal(shape) # [・・・(shot数), 0]はI値、[..., 1]はQ値を格納するための配列
        mean_q = np.zeros(shape)
        mean_q[m2] = self.leak_mean[1] # リーク状態のQ値の平均を設定
        out[..., 1] = mean_q + std * rng.standard_normal(shape)
        return out

    # ---- 2状態の読み出し誤り率 -----------------------------------------------------------

    @cached_property # 計算結果をキャッシュすることで、同じインスタンスで複数回呼び出すときに計算を省略できる
    def _grid(self) -> NDArray:
        return np.linspace(-1.0 - 8.0 * self.sigma, 1.0 + 8.0 * self.sigma, 20001)

    def readout_error_probability(self) -> float:
        """最尤閾値での 2状態分類誤り率 (P(1|0) + P(0|1)) / 2 = ½∫min(f0,f1)dz。

        min(f0,f1) は連続なので台形則の誤差が O(h²) に収まる（領域マスクだと O(h)）。
        グリッド間隔hに対して誤差がh^2で収まる。
        """
        z = self._grid
        f0, f1 = self.pdf_i(z, STATE_0), self.pdf_i(z, STATE_1)
        # 0.5 を掛けるのは、P(1|0) + P(0|1) の平均を取るため。
        # min(f(0),f(1))という関数を、グリッドzに沿って積分することで、2状態分類誤り率を求めることができる。
        return float(0.5 * np.trapezoid(np.minimum(f0, f1), z))
