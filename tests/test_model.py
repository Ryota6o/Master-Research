"""T3-2 受け入れ基準。cwd=paper2/ で `pytest test_model.py`。過学習テストは GPU で約 1 分。"""
import numpy as np
import pytest
import torch

from leakdec.learn.dataset import DataConfig, sample_tokens, token_spec
from leakdec.learn.model import SyndromeTransformer, logical_failures, losses

CFG = DataConfig()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def sampler():
    return CFG.make_sampler()


def _model(spec, **kw):
    return SyndromeTransformer(spec.d_in, spec.qubit_idx, spec.round_idx, **kw)


@pytest.mark.parametrize("rep", ["raw", "soft", "hard"])
def test_forward_shapes_same_code_path(sampler, rep):
    spec = token_spec(rep, sampler.cmap)
    m = _model(spec, d=32, layers=2, heads=2, ff=64)
    b, _ = sample_tokens(sampler, rep, 4, seed=0)
    logical, leak = m(b.x)
    assert logical.shape == (4, 12) and leak.shape == (4, spec.num_tokens)
    loss, l1, l2 = losses(logical, leak, b.logical, b.leak)
    assert torch.isfinite(loss)
    assert logical_failures(logical, b.logical).shape == (4,)


def test_parameter_count_independent_of_single_shot(sampler):
    spec = token_spec("raw", sampler.cmap)
    n_seq = sum(p.numel() for p in _model(spec).parameters())
    n_ss = sum(p.numel() for p in _model(spec, single_shot=True).parameters())
    assert n_seq == n_ss
    assert 0.8e6 < n_seq < 2.0e6


def test_single_shot_leak_logits_ignore_other_rounds(sampler):
    """単発モードでは、他ラウンドのトークンを置換してもリーク logit が変わらない。"""
    spec = token_spec("raw", sampler.cmap)
    torch.manual_seed(0)
    m = _model(spec, d=32, layers=2, heads=2, ff=64, single_shot=True).eval()
    b, _ = sample_tokens(sampler, "raw", 2, seed=1)
    x = b.x.clone()
    _, leak_a = m(x)
    r = torch.as_tensor(spec.round_idx)
    x2 = x.clone()
    x2[:, r != 3] = torch.randn_like(x2[:, r != 3])
    _, leak_b = m(x2)
    torch.testing.assert_close(leak_a[:, r == 3], leak_b[:, r == 3], atol=1e-5, rtol=1e-4)
    # 系列モードでは変わる
    m_seq = _model(spec, d=32, layers=2, heads=2, ff=64).eval()
    m_seq.load_state_dict(m.state_dict())
    _, la = m_seq(x)
    _, lb = m_seq(x2)
    assert not torch.allclose(la[:, r == 3], lb[:, r == 3], atol=1e-3)


def test_overfit_fixed_batch(sampler):
    """R4-3：固定 250 件で loss → 0。表現力と実装の確認。"""
    spec = token_spec("raw", sampler.cmap)
    torch.manual_seed(0)
    m = _model(spec).to(DEV)
    b, _ = sample_tokens(sampler, "raw", 250, seed=7)
    b = b.to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    for step in range(1500):
        logical, leak = m(b.x)
        loss, l1, l2 = losses(logical, leak, b.logical, b.leak)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if loss.item() < 0.01:
            break
    print(f"\n[overfit] step={step} loss={loss.item():.4f} logical={l1.item():.4f} leak={l2.item():.4f}")
    assert loss.item() < 0.01
    assert logical_failures(logical, b.logical).float().mean().item() == 0.0
