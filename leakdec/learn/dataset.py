"""学習用データ：SoftShots → トークン列。3 表現を同じコードパスで切り替える。

  raw  : 補助測定 576 トークン、特徴 (I, Q)。XOR も boxplus もしない
  soft : 検出器 504 トークン、特徴 = boxplus 済み LLR（±20 でクリップし /10）
  hard : 検出器 504 トークン、特徴 = 1 - 2·det ∈ {+1, -1}（LLR の符号と同じ向き）
位置は (補助量子ビット添字 0..71, ラウンド 0..7) の 2 座標で、3 表現で共通。
検出器トークンの位置は構成 2 測定のうち後ろ側のラウンド、リーク教師は 2 測定の OR。

オンライン生成は IterableDataset。DataLoader の各 worker が独立シードで
SoftSyndromeSampler を呼び、毎バッチ新規ショットを返す。
"""
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from leakdec.sim.circuit_map import CircuitMap, default_map_path
from leakdec.sim.gym import QECGym
from leakdec.sim.iq_model import IQModel
from leakdec.sim.leakage_model import LeakageModel
from leakdec.sim.soft_syndrome import SoftShots, SoftSyndromeSampler

Representation = Literal["raw", "soft", "hard"]
LLR_CLIP, LLR_SCALE = 20.0, 10.0


@dataclass(frozen=True)
class TokenSpec:
    representation: str
    qubit_idx: np.ndarray  # (T,) int64
    round_idx: np.ndarray  # (T,) int64
    d_in: int

    @property
    def num_tokens(self) -> int:
        return len(self.qubit_idx)


def token_spec(representation: Representation, cmap: CircuitMap) -> TokenSpec:
    aux = cmap.aux_meas_positions
    if representation == "raw":
        return TokenSpec("raw", cmap.aux_index[aux].astype(np.int64),
                         cmap.meas_round[aux].astype(np.int64), 2)
    if representation in ("soft", "hard"):
        pairs = cmap.detector_aux_measurements()  # (D, 2)、ラウンド昇順
        later = aux[pairs[:, 1]]
        return TokenSpec(representation, cmap.aux_index[later].astype(np.int64),
                         cmap.meas_round[later].astype(np.int64), 1)
    raise ValueError(representation)


@dataclass
class TokenBatch:
    x: torch.Tensor  # (B, T, d_in) float32
    leak: torch.Tensor  # (B, T) float32 ∈ {0,1}
    logical: torch.Tensor  # (B, k) float32 ∈ {0,1}

    # AUDIT(2026-09-16) 【最適化・軽微】DataLoader(pin_memory=True) は dataclass を pin できず素通しにするので、
    #   .to(non_blocking=True) は実際には同期転送になっている。
    # FIX:
    # def pin_memory(self) -> "TokenBatch":
    #     return TokenBatch(self.x.pin_memory(), self.leak.pin_memory(), self.logical.pin_memory())
    def to(self, device) -> "TokenBatch":
        return TokenBatch(self.x.to(device, non_blocking=True),
                          self.leak.to(device, non_blocking=True),
                          self.logical.to(device, non_blocking=True))

    def __len__(self) -> int:
        return self.x.shape[0]


def tokens_from_shots(s: SoftShots, representation: Representation, cmap: CircuitMap) -> TokenBatch:
    if representation == "raw":
        x = s.iq.astype(np.float32)
        leak = s.leaked_aux
    else:
        pairs = cmap.detector_aux_measurements()
        if representation == "soft":
            x = (np.clip(s.detectors_soft, -LLR_CLIP, LLR_CLIP) / LLR_SCALE)[..., None].astype(np.float32)
        else:
            # LLR の符号規約（正 = 検出器 0）に合わせる：det=0 → +1, det=1 → -1
            x = (1.0 - 2.0 * s.detectors_hard.astype(np.float32))[..., None]
        leak = s.leaked_aux[:, pairs[:, 0]] | s.leaked_aux[:, pairs[:, 1]]
    return TokenBatch(torch.from_numpy(np.ascontiguousarray(x)),
                      torch.from_numpy(leak.astype(np.float32)),
                      torch.from_numpy(s.observables.astype(np.float32)))


@dataclass(frozen=True)
class DataConfig:
    code_name: str = "bbc-72-12-6"
    num_rounds: int = 6
    p: float = 0.004
    snr: float = 8.4
    beta: float = 0.05
    p_leak: float = 0.005
    p_seep: float = 0.2
    # AUDIT(2026-09-16) 持続長掃引で ρ を固定する場合の補助（train.py の監査コメント参照）：
    # @staticmethod
    # def p_leak_for(rho: float, p_seep: float) -> float:
    #     return rho * p_seep / (1 - rho + rho * p_seep)
    leak_mean: tuple[float, float] = (0.0, 0.0)
    leak_std: float | None = None

    def make_sampler(self) -> SoftSyndromeSampler:
        gym = QECGym(self.code_name, "X", "circuit", self.p, num_rounds=self.num_rounds,
                     measure_both=True, load_saved_logical_ops=True)
        cmap = CircuitMap.load(default_map_path(self.code_name, "X", self.num_rounds))
        iq = IQModel(snr=self.snr, beta=self.beta, leak_mean=self.leak_mean, leak_std=self.leak_std)
        leak = None if self.p_leak == 0.0 else LeakageModel(self.p_leak, self.p_seep)
        return SoftSyndromeSampler(gym._circuit, iq, leak, cmap)


def sample_tokens(sampler: SoftSyndromeSampler, representation: Representation,
                  shots: int, seed: int) -> tuple[TokenBatch, SoftShots]:
    s = sampler.sample(shots, seed)
    return tokens_from_shots(s, representation, sampler.cmap), s


class OnlineTokenDataset(IterableDataset):
    """毎バッチ新規ショット。worker ごとに SeedSequence(base_seed, worker_id) から系列を切る。"""

    def __init__(self, cfg: DataConfig, representation: Representation,
                 batch_size: int, base_seed: int = 0):
        self.cfg = cfg
        self.representation = representation
        self.batch_size = batch_size
        self.base_seed = base_seed

    def __iter__(self) -> Iterator[TokenBatch]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        sampler = self.cfg.make_sampler()  # fork 後に worker 内で構築する
        seeds = np.random.SeedSequence([self.base_seed, worker_id])
        rng = np.random.default_rng(seeds)
        while True:
            seed = int(rng.integers(0, 2**63 - 1))
            batch, _ = sample_tokens(sampler, self.representation, self.batch_size, seed)
            yield batch


def make_loader(cfg: DataConfig, representation: Representation, batch_size: int,
                num_workers: int = 4, base_seed: int = 0) -> torch.utils.data.DataLoader:
    """DataLoader(batch_size=None)：dataset が既にバッチを返すので自動バッチ化は切る。"""
    ds = OnlineTokenDataset(cfg, representation, batch_size, base_seed)
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=num_workers,
                                       persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers else None,
                                       pin_memory=torch.cuda.is_available())
