"""データ生成パイプライン：Stim 測定 → リーク過程 → IQ サンプル → 3 種の入力表現。

1 ショットから次を同時に作る（§6.1 のアブレーション用。同一ショットであることが要）：
  (i)   detectors_hard : (shots, D)     bool   IQ を閾値判定した硬判定測定の XOR
  (ii)  detectors_soft : (shots, D)     float  測定 LLR を boxplus で潰した検出器 LLR
  (iii) iq             : (shots, A, 2)  float  補助測定ごとの生 IQ 点（潰さない）
教師信号：observables (shots, k)、leaked (shots, R, Q)。

設計メモ
- AUDIT(2026-09-16) 【モデル化の限界・主張の言い回しに影響】最終データ読み出し 72 個には IQ 雑音も測定誤りも
  載せていない（この回路ではデータ読み出しはどの検出器にも寄与せず、ラベルは無雑音値）。実機ではデータ読み出しの
  誤りが論理観測量に直接入り、最終ラウンドの検出器で部分的にしか検出できない。Riverlane はデータ読み出しも soft 化
  している。全デコーダに共通の理想化なので比較は公平だが、「実機相当」と書くときはこの理想化を明記すること。
  また Stim で無雑音の境界ラウンド（0, 7）の補助測定にも IQ 雑音とリークを載せている（DEM に 144 列を追加して整合）。
  リセット（RZ/RX）でリークが解消しない仮定も含め、いずれも plan01 に「置いた仮定」として書く。
- 補助測定 A = 576 のみを soft 化する。最終データ読み出し 72 個は検出器に寄与せず、
  論理観測量の材料（＝ラベル）そのものなので、雑音も掛けず入力にも出さない。
- Stim 回路の測定誤り MZ(p)/MX(p) は残す。IQ の分類誤りはその上に重なる。
  Riverlane §VI D（リセットあり → 追加ノード不要、p' = p_S(1-p) + p(1-p_S)）と同じ構成。
- リーク中の測定は状態 |2> として f^(2) からサンプルし、硬判定は LLR の符号で決まる。
  リークした補助量子ビットが CNOT を通じてデータを汚す効果は対象外。
"""
from dataclasses import dataclass

import numpy as np
import stim

from leakdec.sim.boxplus import detector_llrs
from leakdec.sim.circuit_map import CircuitMap, build_circuit_map
from leakdec.sim.iq_model import STATE_2, IQModel
from leakdec.sim.leakage_model import LeakageModel, expand_to_measurements


@dataclass(frozen=True)
class SoftShots:
    ideal_meas: np.ndarray  # (shots, M) bool  Stim の測定記録（回路雑音込み、IQ 前）
    hard_meas: np.ndarray  # (shots, M) bool  補助は IQ の閾値判定、データは ideal のまま
    detectors_ideal: np.ndarray  # (shots, D) bool  ideal_meas から。既存パイプラインの検出器
    detectors_hard: np.ndarray  # (shots, D) bool  (i)
    detectors_soft: np.ndarray  # (shots, D) float (ii)
    iq: np.ndarray  # (shots, A, 2) float (iii)
    meas_llr: np.ndarray  # (shots, A) float  iq の 2状態 LLR
    observables: np.ndarray  # (shots, k) bool  論理観測量の反転（ラベル）
    leaked: np.ndarray  # (shots, R, Q) bool  ℓ[q, r]
    leaked_aux: np.ndarray  # (shots, A) bool  補助測定ごとの ℓ

    @property
    def shots(self) -> int:
        return self.ideal_meas.shape[0]


class SoftSyndromeSampler:
    def __init__(
        self,
        circuit: stim.Circuit,
        iq_model: IQModel,
        leakage: LeakageModel | None = None,
        cmap: CircuitMap | None = None,
    ):
        self.circuit = circuit
        self.iq_model = iq_model
        self.leakage = leakage
        self.cmap = cmap if cmap is not None else build_circuit_map(circuit)
        # Stim の測定結果から検出器と論理観測量を取り出すためのコンバータ → detectors_ideal (504), observables (12) IQ 前の検出器とラベル
        self._converter = circuit.compile_m2d_converter()

    def _detectors_and_observables(self, meas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        det, obs = self._converter.convert(measurements=meas, separate_observables=True)
        return det, obs

    def sample(self, shots: int, seed: int) -> SoftShots:
        """固定 seed で再現可能。Stim と NumPy の乱数は SeedSequence から分岐させる。"""
        ss = np.random.SeedSequence(seed)
        stim_seed, np_seed = (int(s) for s in ss.generate_state(2, dtype=np.uint64))
        rng = np.random.default_rng(np_seed)
        cmap = self.cmap

        ideal = self.circuit.compile_sampler(seed=stim_seed).sample(shots=shots)
        detectors_ideal, observables = self._detectors_and_observables(ideal)

        if self.leakage is None:
            leaked = np.zeros((shots, cmap.num_aux_rounds, cmap.num_aux_qubits), dtype=np.bool_)
        else:
            # LeakageModel.sample()：→ ℓ[shot, round, aux]、リーク中の測定はstate=2に上書き
            leaked = self.leakage.sample(shots, cmap.num_aux_rounds, cmap.num_aux_qubits, rng)
        leaked_aux = expand_to_measurements(leaked, cmap)[:, cmap.aux_meas_positions] # cmap.aux_meas_positions：補助測定576個だけ取り出す。

        ideal_aux = ideal[:, cmap.aux_meas_positions]
        state = ideal_aux.astype(np.int64)
        state[leaked_aux] = STATE_2
        # IQModel.sample()：→ 補助測定ごとの生 IQ 点 (shots, 576, 2)状態 0/1/2 ごとのガウス（|1> は崩壊込み）
        iq = self.iq_model.sample(state, rng)
        meas_llr = self.iq_model.llr(iq) # → meas_llr (576)

        # 硬判定 = llr < 0 → hard_meas に書き戻し → converter で detectors_hard (i)
        hard = ideal.copy()
        hard[:, cmap.aux_meas_positions] = meas_llr < 0
        detectors_hard, obs_check = self._detectors_and_observables(hard)
        assert np.array_equal(obs_check, observables)  # データ読み出しは触っていない
        # boxplus → detectors_soft (ii)、iq そのものが (iii)
        detectors_soft = detector_llrs(meas_llr, cmap)

        return SoftShots(
            ideal_meas=ideal,
            hard_meas=hard,
            detectors_ideal=detectors_ideal,
            detectors_hard=detectors_hard,
            detectors_soft=detectors_soft,
            iq=iq,
            meas_llr=meas_llr,
            observables=observables,
            leaked=leaked,
            leaked_aux=leaked_aux,
        )
