"""T1-3 受け入れ基準。cwd=paper2/ で `pytest test_leakage_model.py`。"""
import numpy as np
import pytest

from leakdec.sim.circuit_map import KIND_DATA, CircuitMap, default_map_path
from leakdec.sim.leakage_model import LeakageModel, expand_to_measurements


@pytest.fixture(scope="module")
def cmap72():
    return CircuitMap.load(default_map_path("bbc-72-12-6", "X", 6))


def _lag1_conditionals(leaked):
    prev, nxt = leaked[:, :-1], leaked[:, 1:]
    p11 = nxt[prev].mean()
    p10 = nxt[~prev].mean()
    return p11, p10


# ---- 受け入れ基準 1：定常リーク率 -----------------------------------------------------

def test_stationary_rate_matches_formula():
    m = LeakageModel(p_leak=0.02, p_seep=0.2)
    rng = np.random.default_rng(0)
    leaked = m.sample(shots=20_000, num_rounds=8, num_qubits=72, rng=rng)
    per_round = leaked.mean(axis=(0, 2))
    assert np.all(np.abs(per_round - m.stationary_rate) < 2e-3)
    # p_leak ≪ 1 では TODO の式 p_leak/(p_leak+p_seep) と一致（相対差 ~ p_leak）
    naive = 0.02 / (0.02 + 0.2)
    assert abs(m.stationary_rate - naive) / naive < 0.03


# ---- 受け入れ基準 2：平均持続長 -------------------------------------------------------

def _completed_run_lengths(series_2d):
    """(N, T) bool の各行から、両端に触れていない True の連の長さを集める。"""
    lengths = []
    for row in series_2d:
        padded = np.concatenate([[False], row, [False]])
        d = np.diff(padded.astype(np.int8))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        ok = (starts > 0) & (ends < len(row))
        lengths.append(ends[ok] - starts[ok])
    return np.concatenate(lengths)


def test_mean_duration_matches_formula():
    m = LeakageModel(p_leak=0.01, p_seep=0.2)
    rng = np.random.default_rng(1)
    leaked = m.sample(shots=200, num_rounds=3000, num_qubits=10, rng=rng)
    series = leaked.transpose(0, 2, 1).reshape(-1, 3000)
    runs = _completed_run_lengths(series)
    assert len(runs) > 30_000
    assert abs(runs.mean() - m.mean_duration) < 0.1
    assert abs(m.mean_duration - 1 / 0.2) / (1 / 0.2) < 0.02


# ---- 受け入れ基準 3：p_seep = 1 で独立同分布 -------------------------------------------

def test_p_seep_one_is_iid():
    m = LeakageModel(p_leak=0.05, p_seep=1.0)
    rng = np.random.default_rng(2)
    leaked = m.sample(shots=5_000, num_rounds=40, num_qubits=20, rng=rng)
    p11, p10 = _lag1_conditionals(leaked)
    assert abs(p11 - 0.05) < 3e-3
    assert abs(p10 - 0.05) < 3e-3
    assert m.stationary_rate == pytest.approx(0.05)
    assert m.mean_duration == pytest.approx(1 / 0.95)


def test_persistent_leakage_has_positive_lag1_dependence():
    m = LeakageModel(p_leak=0.05, p_seep=0.2)
    rng = np.random.default_rng(3)
    leaked = m.sample(shots=5_000, num_rounds=40, num_qubits=20, rng=rng)
    p11, p10 = _lag1_conditionals(leaked)
    assert abs(p11 - m.p_stay_leaked) < 5e-3
    assert abs(p10 - 0.05) < 3e-3
    assert p11 > 10 * p10


def test_clean_init_starts_at_p_leak():
    m = LeakageModel(p_leak=0.03, p_seep=0.5)
    rng = np.random.default_rng(4)
    leaked = m.sample(shots=20_000, num_rounds=3, num_qubits=50, rng=rng, init="clean")
    assert abs(leaked[:, 0].mean() - 0.03) < 2e-3


# ---- 測定インデックスへの展開 ----------------------------------------------------------

def test_expand_to_measurements_places_flag_on_correct_measurement(cmap72):
    leaked = np.zeros((3, cmap72.num_aux_rounds, cmap72.num_aux_qubits), dtype=np.bool_)
    leaked[1, 3, 5] = True
    out = expand_to_measurements(leaked, cmap72)
    assert out.shape == (3, 648)
    assert out[0].sum() == 0 and out[2].sum() == 0
    flagged = np.nonzero(out[1])[0]
    assert len(flagged) == 1
    m = flagged[0]
    assert cmap72.meas_round[m] == 3
    assert cmap72.meas_qubit[m] == 2 * cmap72.block_size + 5
    assert cmap72.meas_kind[m] != KIND_DATA


def test_expand_never_flags_data_measurements(cmap72):
    leaked = np.ones((2, cmap72.num_aux_rounds, cmap72.num_aux_qubits), dtype=np.bool_)
    out = expand_to_measurements(leaked, cmap72)
    assert np.all(out[:, cmap72.is_aux])
    assert not out[:, ~cmap72.is_aux].any()


def test_circuit_dimensions(cmap72):
    assert cmap72.block_size == 36
    assert cmap72.num_aux_qubits == 72
    assert cmap72.num_aux_rounds == 8
    m = LeakageModel(p_leak=0.01, p_seep=0.2)
    leaked, leaked_meas = m.sample_for_circuit(10, cmap72, np.random.default_rng(0))
    assert leaked.shape == (10, 8, 72)
    assert leaked_meas.shape == (10, 648)
