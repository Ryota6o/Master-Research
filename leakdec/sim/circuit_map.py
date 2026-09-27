"""測定 → 検出器の線形写像 (M2D) と、測定インデックスの (種別, 量子ビット, ラウンド) 対応表。

soft 測定値から soft 検出器を作るとき（boxplus）と、リーク過程を補助量子ビットごとに
載せるときに使う。行列と対応表は Stim 回路そのものから導出し、ハードコードしない。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

KIND_Z_AUX = 0
KIND_X_AUX = 1
KIND_DATA = 2
KIND_NAMES = {KIND_Z_AUX: "Z_aux", KIND_X_AUX: "X_aux", KIND_DATA: "data"}

_MEASUREMENT_GATES = {"M", "MZ", "MX", "MY", "MR", "MRX", "MRY"}


@dataclass(frozen=True)
class CircuitMap:
    m2d: np.ndarray  # (num_detectors, num_measurements) bool。検出器 = 該当測定の XOR
    meas_kind: np.ndarray  # (num_measurements,) int8。KIND_*
    meas_qubit: np.ndarray  # (num_measurements,) int32。Stim の量子ビット番号
    meas_round: np.ndarray  # (num_measurements,) int32。同一量子ビットの何回目の測定か

    @property
    def num_detectors(self) -> int:
        return self.m2d.shape[0]

    @property
    def num_measurements(self) -> int:
        return self.m2d.shape[1]

    @property
    def block_size(self) -> int:
        """l·m。データ量子ビット 2 ブロック（各1回測定）から逆算する。"""
        return int((self.meas_kind == KIND_DATA).sum()) // 2

    @property
    def is_aux(self) -> np.ndarray:
        return self.meas_kind != KIND_DATA

    @property
    def num_aux_qubits(self) -> int:
        return 2 * self.block_size

    @property
    def num_aux_rounds(self) -> int:
        return int(self.meas_round[self.is_aux].max()) + 1

    @property
    def aux_index(self) -> np.ndarray:
        """測定ごとの補助量子ビット添字 0..2·block-1（X 補助が前半、Z 補助が後半）。データ測定は -1。"""
        idx = self.meas_qubit.astype(np.int64) - 2 * self.block_size
        idx[~self.is_aux] = -1
        return idx

    @property
    def num_aux_measurements(self) -> int:
        return int(self.is_aux.sum())

    @property
    def aux_meas_positions(self) -> np.ndarray:
        """補助測定のローカル添字 j → 全測定添字。形状 (num_aux_measurements,)。"""
        return np.nonzero(self.is_aux)[0]

    def detector_aux_measurements(self) -> np.ndarray:
        """`detector_measurements()` を補助測定ローカル添字 (0..num_aux_measurements-1) で返す。"""
        local = np.full(self.num_measurements, -1, dtype=np.int64)
        local[self.is_aux] = np.arange(self.num_aux_measurements)
        pairs = local[self.detector_measurements()]
        if (pairs < 0).any():
            raise ValueError("データ測定に依存する検出器がある")
        return pairs

    def detector_measurements(self) -> np.ndarray:
        """各検出器を構成する測定インデックス (num_detectors, 2)。ラウンド昇順。"""
        rows, cols = np.nonzero(self.m2d)
        counts = np.bincount(rows, minlength=self.num_detectors)
        if not np.all(counts == 2):
            raise ValueError("全検出器がちょうど2測定の XOR である前提が崩れている")
        pairs = cols.reshape(self.num_detectors, 2)
        order = np.argsort(self.meas_round[pairs], axis=1)
        return np.take_along_axis(pairs, order, axis=1)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            m2d=self.m2d,
            meas_kind=self.meas_kind,
            meas_qubit=self.meas_qubit,
            meas_round=self.meas_round,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CircuitMap":
        with np.load(path) as f:
            return cls(
                m2d=f["m2d"],
                meas_kind=f["meas_kind"],
                meas_qubit=f["meas_qubit"],
                meas_round=f["meas_round"],
            )


def extract_m2d(circuit: stim.Circuit) -> np.ndarray:
    """単位行列を m2d コンバータに通して線形写像を取り出す。

    Stim の検出イベントは「測定パリティ XOR 無雑音参照パリティ」なので、
    全0測定を通したときの出力をオフセットとして差し引く。
    """
    converter = circuit.compile_m2d_converter()
    n_meas = circuit.num_measurements
    offset = converter.convert(
        measurements=np.zeros((1, n_meas), dtype=np.bool_), append_observables=False
    )
    if offset.any():
        raise ValueError("無雑音参照パリティが非0の検出器がある（オフセット前提が崩れている）")
    unit = np.eye(n_meas, dtype=np.bool_)
    response = converter.convert(measurements=unit, append_observables=False)
    return response.T.copy()


def extract_measurement_table(circuit: stim.Circuit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """命令列を走査して、測定インデックスごとの (種別, 量子ビット, ラウンド) を返す。

    量子ビットのブロック配置は circuit_gen と同じ:
    [L data | R data | X aux | Z aux]、各ブロック num_qubits // 4 個。
    """
    block = circuit.num_qubits // 4
    if block * 4 != circuit.num_qubits:
        raise ValueError(f"量子ビット数 {circuit.num_qubits} が 4 ブロックに割れない")

    kinds, qubits = [], []
    for inst in circuit.flattened():
        if inst.name not in _MEASUREMENT_GATES:
            if inst.name == "MPP":
                raise NotImplementedError("MPP は非対応")
            continue
        for t in inst.targets_copy():
            q = t.value
            if q < 2 * block:
                kinds.append(KIND_DATA)
            elif q < 3 * block:
                kinds.append(KIND_X_AUX)
            else:
                kinds.append(KIND_Z_AUX)
            qubits.append(q)

    if len(qubits) != circuit.num_measurements:
        raise ValueError(f"走査した測定数 {len(qubits)} が circuit.num_measurements {circuit.num_measurements} と不一致")

    qubits_arr = np.asarray(qubits, dtype=np.int32)
    rounds = np.zeros(len(qubits), dtype=np.int32)
    seen = np.zeros(circuit.num_qubits, dtype=np.int32)
    for i, q in enumerate(qubits_arr):
        rounds[i] = seen[q]
        seen[q] += 1
    return np.asarray(kinds, dtype=np.int8), qubits_arr, rounds


def build_circuit_map(circuit: stim.Circuit) -> CircuitMap:
    kind, qubit, rnd = extract_measurement_table(circuit)
    return CircuitMap(m2d=extract_m2d(circuit), meas_kind=kind, meas_qubit=qubit, meas_round=rnd)


def default_map_path(code_name: str, logical_operator: str, num_rounds: int) -> Path:
    return Path("data/circuit_map") / f"{code_name}_{logical_operator}_r{num_rounds}.npz"


if __name__ == "__main__":
    from leakdec.sim.gym import QECGym

    for code_name, num_rounds in [("bbc-72-12-6", 6), ("bbc-144-12-12", 12)]:
        g = QECGym(code_name, "X", "circuit", 0.006, num_rounds=num_rounds,
                   measure_both=True, load_saved_logical_ops=True)
        cmap = build_circuit_map(g._circuit)
        path = default_map_path(code_name, "X", num_rounds)
        path.parent.mkdir(parents=True, exist_ok=True)
        cmap.save(path)
        counts = {KIND_NAMES[k]: int((cmap.meas_kind == k).sum()) for k in KIND_NAMES}
        print(f"{code_name}: m2d={cmap.m2d.shape}, {counts} -> {path}")
