"""T3-1 受け入れ基準。cwd=paper2/ で `pytest test_dataset.py`。"""
import time

import numpy as np
import pytest
import torch

from leakdec.learn.dataset import DataConfig, make_loader, sample_tokens, token_spec

CFG = DataConfig()


@pytest.fixture(scope="module")
def sampler():
    return CFG.make_sampler()


@pytest.mark.parametrize("rep,T,d_in", [("raw", 576, 2), ("soft", 504, 1), ("hard", 504, 1)])
def test_token_spec_and_shapes(sampler, rep, T, d_in):
    spec = token_spec(rep, sampler.cmap)
    assert spec.num_tokens == T and spec.d_in == d_in
    assert spec.qubit_idx.min() == 0 and spec.qubit_idx.max() == 71
    assert set(np.unique(spec.round_idx)) == (set(range(8)) if rep == "raw" else set(range(1, 8)))
    b, s = sample_tokens(sampler, rep, 16, seed=0)
    assert b.x.shape == (16, T, d_in) and b.x.dtype == torch.float32
    assert b.leak.shape == (16, T) and b.logical.shape == (16, 12)


def test_three_representations_same_shot(sampler):
    braw, sraw = sample_tokens(sampler, "raw", 32, seed=5)
    bsoft, ssoft = sample_tokens(sampler, "soft", 32, seed=5)
    bhard, shard = sample_tokens(sampler, "hard", 32, seed=5)
    np.testing.assert_array_equal(sraw.iq, ssoft.iq)
    np.testing.assert_array_equal(sraw.iq, shard.iq)
    torch.testing.assert_close(braw.logical, bsoft.logical)
    torch.testing.assert_close(braw.logical, bhard.logical)
    # hard 特徴の符号 = soft 特徴の符号（同じ検出器）
    assert torch.equal(bhard.x[..., 0] < 0, bsoft.x[..., 0] < 0)


def test_detector_leak_label_is_or_of_two_measurements(sampler):
    b, s = sample_tokens(sampler, "soft", 64, seed=1)
    pairs = sampler.cmap.detector_aux_measurements()
    expect = s.leaked_aux[:, pairs[:, 0]] | s.leaked_aux[:, pairs[:, 1]]
    np.testing.assert_array_equal(b.leak.numpy().astype(bool), expect)


def test_soft_feature_scaling(sampler):
    b, _ = sample_tokens(sampler, "soft", 64, seed=2)
    assert b.x.abs().max() <= 2.0


def test_online_loader_distinct_batches_and_throughput():
    loader = make_loader(CFG, "raw", batch_size=256, num_workers=2, base_seed=3)
    it = iter(loader)
    a = next(it)  # worker の立ち上げを含むので計測から外す
    t0 = time.perf_counter()
    n = 6
    batches = [next(it) for _ in range(n)]
    dt = time.perf_counter() - t0
    shots_per_s = n * 256 / dt
    print(f"\n[throughput] online raw: {shots_per_s:.0f} shots/s with 2 workers")
    assert a.x.shape == (256, 576, 2)
    assert not torch.equal(batches[0].x, batches[1].x)
    assert shots_per_s > 2000  # 学習側 256×15 it/s ≈ 4000 shots/s に 4 worker で届く水準


def test_144_code_token_specs():
    from leakdec.sim.circuit_map import CircuitMap, default_map_path
    from leakdec.learn.model import SyndromeTransformer
    cmap = CircuitMap.load(default_map_path("bbc-144-12-12", "X", 12))
    assert cmap.num_aux_qubits == 144 and cmap.num_aux_rounds == 14
    raw, soft = token_spec("raw", cmap), token_spec("soft", cmap)
    assert raw.num_tokens == 144 * 14 and soft.num_tokens == 144 * 13
    assert raw.qubit_idx.max() == 143 and raw.round_idx.max() == 13
    m = SyndromeTransformer(2, raw.qubit_idx, raw.round_idx, num_qubits=144, num_rounds=14,
                            d=16, layers=1, heads=2, ff=32)
    logical, leak = m(torch.zeros(2, raw.num_tokens, 2))
    assert logical.shape == (2, 12) and leak.shape == (2, raw.num_tokens)
