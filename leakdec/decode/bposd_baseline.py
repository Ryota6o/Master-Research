"""BP-OSD ベースライン（B0: hard / B1: soft）。

soft 情報の入れ方は Riverlane (arXiv:2504.03504) §VI D・式(19)：
補助測定 m の分類誤り確率 p_S,m で、対応する誤り機構の事前確率を
    p' = p_S (1 - p) + p (1 - p_S)
に更新する。リセットありの回路では測定分類誤りは「その測定の前のビット反転」と同じ
検出器を叩くので、内側ラウンドの測定には DEM に既に列がある（追加ノード不要）。
ただし境界ラウンド（無雑音の初期・最終ラウンド）の測定は DEM に列がないので、
その測定が触る単一検出器の列を H に**追加**する。追加しないと B0/B1 とも誤特定になる。

B0 は p_S を IQ モデルの平均分類誤り率（定数）に、B1 はショットごとの sigmoid(-|LLR|) にする。
B0 の事前確率も同じ式で更新するのは、hard デコーダを弱くしないため（RULES R1）。

必ず schedule="serial"（R1-1）。ワーカはパスごとにデコーダ1個（R2-1）。
"""
import time
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from ldpc import BpOsdDecoder
from scipy.sparse import csr_matrix

from leakdec.sim.circuit_map import CircuitMap


@dataclass(frozen=True)
class SoftDecodingProblem:
    H: np.ndarray  # (D, N') bool
    L: np.ndarray  # (k, N') bool
    priors: np.ndarray  # (N',) float。追加列は 0（常に p_S で更新される前提）
    meas_col: np.ndarray  # (A,) int。補助測定 j ↔ 列
    num_added: int


def build_soft_decoding_problem(H: np.ndarray, L: np.ndarray, priors: np.ndarray,
                                cmap: CircuitMap) -> SoftDecodingProblem:
    H = np.asarray(H, dtype=np.bool_)
    L = np.asarray(L, dtype=np.bool_)
    D, N = H.shape

    # 観測量を反転しない列を検出器シグネチャで引く。
    # DEM には REPEAT 境界でマージされない重複列（重み 3, 6）があるが測定シグネチャとは無関係。
    sig_to_cols: dict[tuple[int, ...], list[int]] = {}
    obs_free = ~L.any(axis=0)
    for c in np.nonzero(obs_free)[0]:
        sig = tuple(np.nonzero(H[:, c])[0].tolist())
        sig_to_cols.setdefault(sig, []).append(int(c))

    m2d_aux = cmap.m2d[:, cmap.aux_meas_positions]
    meas_col = np.empty(cmap.num_aux_measurements, dtype=np.int64)
    new_cols = []
    for j in range(cmap.num_aux_measurements):
        sig = tuple(np.nonzero(m2d_aux[:, j])[0].tolist())
        cols = sig_to_cols.get(sig, [])
        if len(cols) > 1:
            raise ValueError(f"補助測定 j={j} のシグネチャ {sig} に DEM 列が複数ある: {cols}")
        if cols:
            meas_col[j] = cols[0]
        else:
            if len(sig) != 1:
                raise ValueError(f"境界以外で DEM に列がない補助測定がある: j={j}, sig={sig}")
            meas_col[j] = N + len(new_cols)
            new_cols.append(sig[0])

    if new_cols:
        add = np.zeros((D, len(new_cols)), dtype=np.bool_)
        add[new_cols, np.arange(len(new_cols))] = True
        H_aug = np.concatenate([H, add], axis=1)
        L_aug = np.concatenate([L, np.zeros((L.shape[0], len(new_cols)), dtype=np.bool_)], axis=1)
        priors_aug = np.concatenate([np.asarray(priors, dtype=np.float64), np.zeros(len(new_cols))])
    else:
        H_aug, L_aug, priors_aug = H, L, np.asarray(priors, dtype=np.float64)
    return SoftDecodingProblem(H_aug, L_aug, priors_aug, meas_col, len(new_cols))


def classification_error_from_llr(llr: np.ndarray) -> np.ndarray:
    """等事前確率の 2 状態 LLR から硬判定の誤り確率 P(誤り | z) = 1/(1+e^{|L|})。"""
    # AUDIT(2026-09-16) 【前提の注記】等事前確率（P(0)=P(1)=½）の事後確率。実際の補助測定は X 補助が 1 の割合 25%、
    #   Z 補助が 50% で偏っており、実測の平均分類誤り率は 0.0297（モデルの p_S_avg 0.0319 は ½∫min(f0,f1)）。
    #   Riverlane も等事前確率で扱っているので文献と同じ扱いだが、B0/B1 がわずかに誤特定である点は TODO に記録する。
    return 1.0 / (1.0 + np.exp(np.abs(llr)))


def update_priors(problem: SoftDecodingProblem, p_s: np.ndarray) -> np.ndarray:
    """p_s: (A,) または (S, A)。返り値は (N',) または (S, N')。式(19)。"""
    p_s = np.asarray(p_s, dtype=np.float64)
    base = problem.priors
    out = np.broadcast_to(base, p_s.shape[:-1] + base.shape).copy()
    p = base[problem.meas_col]
    out[..., problem.meas_col] = p_s * (1.0 - p) + p * (1.0 - p_s)
    return out


# ---- デコーダ ---------------------------------------------------------------------------

def make_decoder(problem: SoftDecodingProblem, max_iter: int = 200, osd_order: int = 7,
                 omp_threads: int = 1) -> BpOsdDecoder:
    dec = BpOsdDecoder(
        csr_matrix(problem.H.astype(np.uint8)),
        error_channel=list(np.clip(problem.priors, 1e-12, 1 - 1e-12)),
        bp_method="minimum_sum",
        ms_scaling_factor=0.625,
        max_iter=max_iter,
        schedule="serial",
        osd_method="osd_cs",
        osd_order=osd_order,
    )
    dec.omp_thread_count = omp_threads
    return dec


_W: dict = {}


def _init_worker(problem, max_iter, osd_order):
    _W["problem"] = problem
    _W["dec"] = make_decoder(problem, max_iter, osd_order)


def _decode_chunk(task):
    idxs, syndromes, priors = task
    dec, L = _W["dec"], _W["problem"].L
    pred = np.zeros((len(idxs), L.shape[0]), dtype=np.bool_)
    converged = np.zeros(len(idxs), dtype=np.bool_)
    seconds = np.zeros(len(idxs), dtype=np.float64)
    for j in range(len(idxs)):
        t0 = time.perf_counter()
        # AUDIT(2026-09-16) 【計測の妥当性】この秒数は「n_workers 個のプロセスが同時に走っている状態」での壁時計。
        #   T1-6 の単発計測 ≈1.2 s に対し T2-2（22 ワーカ同時）は中央値 6.3 s と 5 倍違う。メモリ帯域と CPU の競合が
        #   乗っているので、論文に出すレイテンシは n_workers=1・他ジョブ無しで 100〜200 ショットを別途測ること。
        #   また T2-2 では p99/中央値 = 1.07 で裾が無い（BP 収束 ≤10% で OSD の固定コストが支配）。
        #   plan01 §0.2/§3.6・RULES R3-5 の「p99 = 114 s、データ依存で発散」は本設定では再現していない。
        # FIX（別関数として追加する案）:
        # def measure_latency(problem, detectors, priors=None, n=200, **kw):
        #     """単一プロセス・逐次で n ショットを復号し秒数 (n,) を返す。他ジョブが無い状態で呼ぶこと。"""
        #     _init_worker(problem, kw.get("max_iter", 200), kw.get("osd_order", 7))
        #     _, _, _, secs = _decode_chunk((np.arange(n), detectors[:n], None if priors is None else priors[:n]))
        #     return secs
        # soft 化のショットごとのコスト（事前確率更新）も復号時間に含める
        if priors is not None:
            dec.update_channel_probs(np.clip(priors[j], 1e-12, 1 - 1e-12))
        e = dec.decode(syndromes[j].astype(np.uint8))
        seconds[j] = time.perf_counter() - t0
        # AUDIT(2026-09-16) 【最適化・軽微】L.astype を毎ショット呼んでいる（12×19152 のコピー）。復号 6 s に対し無視できるが無駄。
        # FIX: _init_worker で _W["L8"] = problem.L.astype(np.uint8) を一度作り、ここでは _W["L8"] @ e を使う。
        pred[j] = (L.astype(np.uint8) @ e) % 2
        converged[j] = bool(dec.converge)
    return idxs, pred, converged, seconds


def decode_shots(problem: SoftDecodingProblem, detectors: np.ndarray,
                 priors: np.ndarray | None = None, *, n_workers: int = 1,
                 chunk: int = 8, max_iter: int = 200, osd_order: int = 7):
    """detectors (S, D) を復号し、予測観測量 (S, k)、BP 収束フラグ (S,)、復号時間 (S,) [s] を返す。

    priors=None なら problem.priors（静的）。(S, N') ならショットごとに更新（B1）。
    静的な p_S 更新（B0）は呼び出し側で problem を差し替える（`with_static_priors`）。
    """
    S = detectors.shape[0]
    tasks = [(np.arange(i, min(i + chunk, S)),
              detectors[i:i + chunk],
              None if priors is None else priors[i:i + chunk])
             for i in range(0, S, chunk)]
    pred = np.zeros((S, problem.L.shape[0]), dtype=np.bool_)
    conv = np.zeros(S, dtype=np.bool_)
    secs = np.zeros(S, dtype=np.float64)
    if n_workers == 1:
        _init_worker(problem, max_iter, osd_order)
        results = map(_decode_chunk, tasks)
    else:
        pool = Pool(n_workers, initializer=_init_worker, initargs=(problem, max_iter, osd_order))
        results = pool.imap_unordered(_decode_chunk, tasks)
    for idxs, p, c, s in results:
        pred[idxs] = p
        conv[idxs] = c
        secs[idxs] = s
    if n_workers != 1:
        pool.close()
        pool.join()
    return pred, conv, secs


def with_static_priors(problem: SoftDecodingProblem, p_s_avg: float) -> SoftDecodingProblem:
    """B0 用：全補助測定に定数 p_S を入れた事前確率を持つ問題を返す。"""
    priors = update_priors(problem, np.full(len(problem.meas_col), p_s_avg))
    return SoftDecodingProblem(problem.H, problem.L, priors, problem.meas_col, problem.num_added)
