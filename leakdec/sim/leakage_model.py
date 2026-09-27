"""補助量子ビットのリーク持続過程（2状態マルコフ連鎖）。

各補助量子ビットはラウンドごとに
  1. リーク中なら確率 p_seep で計算部分空間へ復帰し、
  2. （復帰直後を含め）非リークなら確率 p_leak でリークする。
潜在変数 ℓ[shot, round, aux] = そのラウンドの測定時にリークしていたか。教師信号として返す。

復帰と再リークを同一ラウンド内で許すのは、p_seep = 1 で ℓ がラウンド間で厳密に独立
（i.i.d. Bernoulli(p_leak)）になるようにするため。これが機構証明（plan01 §6.3）の対照条件。
復帰後に再リークを禁じる定義だと p_seep = 1 でも「リークの翌ラウンドは必ず正常」という
負の相関が残り、対照条件が汚れる。

その代わり定常率と平均持続長には (1 - p_leak) の補正がつく：
  定常率      = p_leak / (p_leak + p_seep (1 - p_leak))
  平均持続長  = 1 / (p_seep (1 - p_leak))
p_leak ≪ 1 では TODO の p_leak/(p_leak+p_seep)、1/p_seep に一致する。

データ量子ビットのリークはこのモデルの対象外（測定側の効果のみを扱う。plan01 §4）。
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from leakdec.sim.circuit_map import CircuitMap


@dataclass(frozen=True) # flozen = True：一度作ったら値を変更できないという意味
class LeakageModel:
    p_leak: float # 正常な状態からリークする確率
    p_seep: float # リーク状態から正常に戻る確率

    def __post_init__(self):
        if not (0.0 <= self.p_leak <= 1.0 and 0.0 < self.p_seep <= 1.0):
            raise ValueError("0 <= p_leak <= 1, 0 < p_seep <= 1 が必要")

    @property # 正常になってから同じラウンド内でもう一度p_leakでリークする確率を返す
    def p_stay_leaked(self) -> float:
        """P(ℓ_r = 1 | ℓ_{r-1} = 1)"""
        return 1.0 - self.p_seep * (1.0 - self.p_leak)

    @property
    def stationary_rate(self) -> float:
        return self.p_leak / (self.p_leak + self.p_seep * (1.0 - self.p_leak))

    @property
    def mean_duration(self) -> float:
        """連続してリークしているラウンド数の期待値。"""
        return 1.0 / (self.p_seep * (1.0 - self.p_leak))

    def sample(
        self,
        shots: int,
        num_rounds: int,
        num_qubits: int,
        rng: np.random.Generator,
        init: Literal["stationary", "clean"] = "stationary",
    ) -> np.ndarray:
        """ℓ (shots, num_rounds, num_qubits) bool を引く。

        init="stationary" なら全ラウンドの周辺分布が定常率。"clean" は全量子ビットが
        非リークから始まり、ラウンド 0 の遷移から p_leak でリークしうる。
        """
        if init == "stationary":
            cur = rng.random((shots, num_qubits)) < self.stationary_rate
        elif init == "clean":
            cur = np.zeros((shots, num_qubits), dtype=np.bool_)
        else:
            raise ValueError(init)

        out = np.empty((shots, num_rounds, num_qubits), dtype=np.bool_)
        for r in range(num_rounds): # 定常状態から始めることと、全員リーク無しから始めるかが決まる。
            u = rng.random((shots, num_qubits))
            cur = np.where(cur, u < self.p_stay_leaked, u < self.p_leak) # np.whereで全ショット・全量子ビット分まとめて一度に計算している。
            out[:, r] = cur
        return out

    # (shots, ラウンド数, 補助量子ビット数) という「抽象的な」形の配列
    def sample_for_circuit(
        self, shots: int, cmap: CircuitMap, rng: np.random.Generator,
        init: Literal["stationary", "clean"] = "stationary",
    ) -> tuple[np.ndarray, np.ndarray]:
        """回路の (ラウンド数, 補助量子ビット数) で ℓ を引き、測定インデックス展開も返す。

        Returns:
            leaked: (shots, num_aux_rounds, num_aux_qubits)
            leaked_meas: (shots, num_measurements)。データ測定は常に False
        """
        leaked = self.sample(shots, cmap.num_aux_rounds, cmap.num_aux_qubits, rng, init)
        return leaked, expand_to_measurements(leaked, cmap)

# どの測定インデックスが、どのラウンド・どの補助量子ビットに対応するか
def expand_to_measurements(leaked: np.ndarray, cmap: CircuitMap) -> np.ndarray:
    """ℓ (shots, rounds, aux) を測定インデックス順 (shots, num_measurements) に並べ替える。"""
    aux = cmap.is_aux
    out = np.zeros((leaked.shape[0], cmap.num_measurements), dtype=np.bool_)
    out[:, aux] = leaked[:, cmap.meas_round[aux], cmap.aux_index[aux]]
    return out
