"""
head-to-head 用の「共有シンドローム集合」ユーティリティ

QECGym は生成のたびに stim のシードを randint で引き直すため、スクリプトを別々に
走らせると拡散デコーダと BP-OSD 系ベースラインが別のシンドロームを見てしまう。
post-selection の比較は「同じショット集合上での棄却率 vs 残留 LER」でなければ
意味がないので、ショットを一度だけ生成して npz に固定し、全スクリプトから
同じファイルを読む。

使い方:
    from leakdec.sim.shots import make_gym, load_or_make_shots
    gym = make_gym(72, 0.006)
    syndro, logerr = load_or_make_shots(gym, 72, 0.006, 2000)
        syndro: (n, num_detectors) uint8
        logerr: (n, num_logical_obs) uint8
"""
import os

import numpy as np

from leakdec.sim.gym import QECGym

SHOT_DIR = "saved_results/shots"


def make_gym(N: int, physical_error_rate: float) -> QECGym:
    """eval_*.py 各スクリプトと同一設定の QECGym を返す。"""
    if N == 72:
        return QECGym("bbc-72-12-6", "X", "circuit",
                      physical_error_rate=physical_error_rate,
                      num_rounds=6, measure_both=True, load_saved_logical_ops=True)
    elif N == 144:
        return QECGym("bbc-144-12-12", "X", "circuit",
                      physical_error_rate=physical_error_rate,
                      num_rounds=12, measure_both=True, load_saved_logical_ops=True)
    raise ValueError(f"Unsupported N={N}")


def shot_path(N: int, physical_error_rate: float, n: int) -> str:
    return os.path.join(SHOT_DIR, f"shots_N{N}_p{physical_error_rate}_n{n}.npz")


def load_or_make_shots(gym: QECGym, N: int, physical_error_rate: float, n: int):
    """
    固定ショット集合を読み込む。無ければ生成して保存する。

    Returns:
        syndro: (n, num_detectors) uint8   検出器データ
        logerr: (n, num_logical_obs) uint8 真の論理反転
    """
    path = shot_path(N, physical_error_rate, n)
    if os.path.exists(path):
        d = np.load(path)
        print(f"[shots] loaded {path}  ({d['syndro'].shape[0]} shots)")
        return d["syndro"], d["logerr"]

    os.makedirs(SHOT_DIR, exist_ok=True)
    syndro, logerr, _ = gym.get_decoding_instances(n, return_errors=False)
    syndro = np.asarray(syndro).astype(np.uint8)
    logerr = np.asarray(logerr).astype(np.uint8)
    np.savez_compressed(path, syndro=syndro, logerr=logerr)
    print(f"[shots] generated and saved {path}  ({n} shots)")
    return syndro, logerr
