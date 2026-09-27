"""T1-1 受け入れ基準。cwd=paper2/ で `pytest test_circuit_map.py` を実行する。"""
import numpy as np
import pytest

from leakdec.sim.circuit_map import (
    KIND_DATA,
    KIND_X_AUX,
    KIND_Z_AUX,
    build_circuit_map,
    extract_m2d,
)
from leakdec.sim.gym import QECGym


@pytest.fixture(scope="module")
def gym72():
    return QECGym("bbc-72-12-6", "X", "circuit", 0.006, num_rounds=6,
                  measure_both=True, load_saved_logical_ops=True)


@pytest.fixture(scope="module")
def cmap72(gym72):
    return build_circuit_map(gym72._circuit)


def test_shape(cmap72):
    assert cmap72.m2d.shape == (504, 648)


def test_row_weight_exactly_2_col_weight_at_most_2(cmap72):
    row_w = cmap72.m2d.sum(axis=1)
    col_w = cmap72.m2d.sum(axis=0)
    assert np.all(row_w == 2), f"行重み: {np.unique(row_w)}"
    assert col_w.max() <= 2, f"列重み最大: {col_w.max()}"


def test_zero_measurements_give_zero_detectors(gym72):
    conv = gym72._circuit.compile_m2d_converter()
    out = conv.convert(measurements=np.zeros((1, 648), dtype=np.bool_),
                       append_observables=False)
    assert not out.any()


def test_measurement_breakdown_matches_instruction_stream(cmap72):
    assert (cmap72.meas_kind == KIND_Z_AUX).sum() == 288
    assert (cmap72.meas_kind == KIND_X_AUX).sum() == 288
    assert (cmap72.meas_kind == KIND_DATA).sum() == 72


def test_aux_rounds_and_data_round(cmap72):
    aux = cmap72.meas_kind != KIND_DATA
    assert set(np.unique(cmap72.meas_round[aux])) == set(range(8))
    assert np.all(cmap72.meas_round[~aux] == 0)


def test_each_detector_compares_same_aux_qubit_in_consecutive_rounds(cmap72):
    pairs = cmap72.detector_measurements()
    k0, k1 = cmap72.meas_kind[pairs[:, 0]], cmap72.meas_kind[pairs[:, 1]]
    q0, q1 = cmap72.meas_qubit[pairs[:, 0]], cmap72.meas_qubit[pairs[:, 1]]
    r0, r1 = cmap72.meas_round[pairs[:, 0]], cmap72.meas_round[pairs[:, 1]]
    assert np.all(k0 == k1) and np.all(k0 != KIND_DATA)
    assert np.all(q0 == q1)
    assert np.all(r1 - r0 == 1)


def test_data_measurements_touch_no_detector(cmap72):
    data_cols = cmap72.meas_kind == KIND_DATA
    assert not cmap72.m2d[:, data_cols].any()


def test_m2d_reproduces_converter_on_random_input(gym72, cmap72):
    rng = np.random.default_rng(0)
    meas = rng.integers(0, 2, size=(200, 648), dtype=np.uint8).astype(np.bool_)
    conv = gym72._circuit.compile_m2d_converter()
    expected = conv.convert(measurements=meas, append_observables=False)
    got = (meas.astype(np.uint8) @ cmap72.m2d.T.astype(np.uint8)) % 2
    assert np.array_equal(got.astype(np.bool_), expected)


def test_m2d_matches_converter_on_circuit_samples(gym72, cmap72):
    """雑音入り回路サンプル（現実的な測定パターン）でも M2D が検出器を再現する。"""
    meas = gym72._circuit.compile_sampler(seed=1).sample(shots=200)
    got = (meas.astype(np.uint8) @ cmap72.m2d.T.astype(np.uint8)) % 2
    conv = gym72._circuit.compile_m2d_converter()
    expected = conv.convert(measurements=meas, append_observables=False)
    assert np.array_equal(got.astype(np.bool_), expected)


def test_144_code_shape():
    g = QECGym("bbc-144-12-12", "X", "circuit", 0.006, num_rounds=12,
               measure_both=True, load_saved_logical_ops=True)
    cmap = build_circuit_map(g._circuit)
    assert cmap.m2d.shape == (144 * 13, 144 * 14 + 144)
    assert np.all(cmap.m2d.sum(axis=1) == 2)
    assert cmap.m2d.sum(axis=0).max() <= 2


def test_save_load_roundtrip(cmap72, tmp_path):
    p = tmp_path / "m.npz"
    cmap72.save(p)
    from leakdec.sim.circuit_map import CircuitMap
    back = CircuitMap.load(p)
    assert np.array_equal(back.m2d, cmap72.m2d)
    assert np.array_equal(back.meas_kind, cmap72.meas_kind)
    assert np.array_equal(back.meas_qubit, cmap72.meas_qubit)
    assert np.array_equal(back.meas_round, cmap72.meas_round)


def test_extract_m2d_rejects_nonzero_offset():
    import stim
    c = stim.Circuit("""
        R 0
        X 0
        M 0
        DETECTOR rec[-1]
    """)
    with pytest.raises(ValueError):
        extract_m2d(c)
