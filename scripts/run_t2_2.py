"""T2-2：リークありで B0 / B1 / B2 / B3 / B3s を同一ショットで比較する。

    nohup python -u scripts/run_t2_2.py [shots] [workers] [p_leak] [p_seep] > results/t2_2_<tag>.log 2>&1 &

  B0  hard 検出器 + 平均 p_S で更新した静的事前確率
  B1  soft：ショットごとの 2 状態 p_S
  B2  B1 + Hanisch 流（円形境界 1% 外れ値 → p_S = 0.5）※参考値
  B3  B1 + 最適 per-shot 3 状態判別器 → p_S = 0.5（plan の「本当の対戦相手」）
  B3s B1 + Bayes 最適 per-shot p_S（判別せず事後確率を直接使う。per-shot の上限）
  B4  B1 + HMM（ラウンド方向の前向き後向き）でリーク事後確率 → B3s と同じ式で p_S（持続性を使う古典の上限）
  ORC オラクル：真の ℓ でリーク中の測定だけ p_S = 0.5（古典デコーダにリーク推論を渡す上限）
受け入れ：B1 > B0、B3 ≥ B2、レイテンシ中央値と p99 を記録。

第 5 引数で条件をカンマ区切りで絞れる（例 `ORC`）。同じ seed なのでショットは同一で、
既存の全条件の npz があれば B1 との対応のある比較も出す。
"""
import sys
import time
from pathlib import Path

import numpy as np

from leakdec.decode.bposd_baseline import (build_soft_decoding_problem, classification_error_from_llr,
                            decode_shots, update_priors, with_static_priors)
from leakdec.sim.circuit_map import CircuitMap, default_map_path
from leakdec.sim.gym import QECGym
from leakdec.sim.iq_model import IQModel
from leakdec.decode.leak_baselines import (FlagStats, bayes_optimal_p_s, hanisch_outlier_flags,
                            optimal_leak_flags, p_s_with_leak_flags)
from leakdec.decode.leak_hmm import hmm_leak_posterior, hmm_p_s
from leakdec.sim.leakage_model import LeakageModel
from leakdec.sim.soft_syndrome import SoftSyndromeSampler
from leakdec.analysis.stats import latency_line, ler_line, mcnemar_counts

P = 0.004
SNR, BETA = 8.4, 0.05
SHOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 22
# p_leak=0.02 だと補助測定の 9% がリークで LER 0.4〜0.7（符号崩壊、比較にならない）。
# 0.005 で 2.5%。リーク率の掃引は T4-5
P_LEAK = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
P_SEEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
SEED = 0
TAG = f"pl{P_LEAK}_ps{P_SEEP}"
ALL = ("B0", "B1", "B2", "B3", "B3s", "B4", "ORC")
ONLY = tuple(sys.argv[5].split(",")) if len(sys.argv) > 5 else ALL


def main():
    Path("results").mkdir(exist_ok=True)
    # AUDIT(2026-09-16) 【拡張性】符号とラウンド数が固定。144 符号の T2-2 を回すには引数化が要る（train/evaluate は --code 対応済み）。
    #   また TAG（結果ファイル名）に符号が入っていないので、144 で回すと 72 の結果を上書きする。
    # FIX:
    # CODE = sys.argv[6] if len(sys.argv) > 6 else "bbc-72-12-6"; ROUNDS = {"bbc-72-12-6": 6, "bbc-144-12-12": 12}[CODE]
    # TAG = f"{CODE}_pl{P_LEAK}_ps{P_SEEP}"
    gym = QECGym("bbc-72-12-6", "X", "circuit", P, num_rounds=6,
                 measure_both=True, load_saved_logical_ops=True)
    cmap = CircuitMap.load(default_map_path("bbc-72-12-6", "X", 6))
    iq = IQModel(snr=SNR, beta=BETA)
    leak = LeakageModel(p_leak=P_LEAK, p_seep=P_SEEP)
    rho = leak.stationary_rate
    p_s_avg = iq.readout_error_probability()
    print(f"[config] p={P} snr={SNR} beta={BETA} p_S_avg={p_s_avg:.4f} "
          f"p_leak={P_LEAK} p_seep={P_SEEP} rho={rho:.4f} mean_dur={leak.mean_duration:.2f} "
          f"shots={SHOTS} workers={WORKERS}", flush=True)

    s = SoftSyndromeSampler(gym._circuit, iq, leak, cmap).sample(SHOTS, seed=SEED)
    print(f"[data] det rate ideal={s.detectors_ideal.mean():.4f} hard={s.detectors_hard.mean():.4f}  "
          f"leaked aux meas={s.leaked_aux.mean():.4f}", flush=True)

    problem = build_soft_decoding_problem(
        gym.get_spacetime_parity_check_matrix(), gym.get_logical_decoding_matrix(),
        gym.get_channel_probabalities(), cmap)

    flags_b2 = hanisch_outlier_flags(s.iq, iq)
    flags_b3 = optimal_leak_flags(s.iq, iq, rho)
    st2, st3 = FlagStats.of(flags_b2, s.leaked_aux), FlagStats.of(flags_b3, s.leaked_aux)
    print(f"[flags] B2 precision={st2.precision:.3f} recall={st2.recall:.3f} flagged={st2.flagged_rate:.4f}", flush=True)
    print(f"[flags] B3 precision={st3.precision:.3f} recall={st3.recall:.3f} flagged={st3.flagged_rate:.4f}", flush=True)

    p_s_b1 = classification_error_from_llr(s.meas_llr)
    post_b4 = hmm_leak_posterior(s.iq, iq, leak, cmap)
    st4 = FlagStats.of(post_b4 > 0.5, s.leaked_aux)
    print(f"[flags] B4 (HMM, 閾値 0.5) precision={st4.precision:.3f} recall={st4.recall:.3f} flagged={st4.flagged_rate:.4f}", flush=True)
    conditions = {
        "B0": (with_static_priors(problem, p_s_avg), None),
        "B1": (problem, update_priors(problem, p_s_b1)),
        "B2": (problem, update_priors(problem, p_s_with_leak_flags(s.meas_llr, flags_b2))),
        "B3": (problem, update_priors(problem, p_s_with_leak_flags(s.meas_llr, flags_b3))),
        "B3s": (problem, update_priors(problem, bayes_optimal_p_s(s.iq, iq, rho))),
        "B4": (problem, update_priors(problem, hmm_p_s(s.meas_llr, post_b4))),
        "ORC": (problem, update_priors(problem, p_s_with_leak_flags(s.meas_llr, s.leaked_aux))),
    }

    truth = s.observables
    fails, secs = {}, {}
    full = Path(f"results/t2_2_{TAG}.npz")
    if ONLY != ALL and full.exists():
        prev = np.load(full)
        for k in prev.files:
            if k.startswith("fail_"):
                fails[k[5:]] = prev[k]
    for name, (prob, priors) in conditions.items():
        if name not in ONLY:
            continue
        t = time.time()
        pred, conv, sec = decode_shots(prob, s.detectors_hard, priors, n_workers=WORKERS)
        fails[name] = (pred != truth).any(axis=1)
        secs[name] = sec
        print(f"[{name:3s}] {time.time() - t:.0f}s  BP conv={conv.mean():.3f}  "
              f"{ler_line('LER', int(fails[name].sum()), SHOTS)}  latency {latency_line(sec)}", flush=True)

    print("[pairs] 対応のある比較（左だけ失敗, 右だけ失敗）", flush=True)
    for a, b in (("B0", "B1"), ("B1", "B2"), ("B1", "B3"), ("B2", "B3"), ("B3", "B3s"), ("B1", "B3s"),
                 ("B1", "B4"), ("B3s", "B4"), ("B4", "ORC"), ("B1", "ORC"), ("B3s", "ORC")):
        if a in fails and b in fails:
            print(f"  {a} vs {b}: {mcnemar_counts(fails[a], fails[b])}", flush=True)

    if {"B0", "B1", "B2", "B3"} <= fails.keys():
        ok1 = fails["B1"].sum() < fails["B0"].sum()
        ok2 = fails["B3"].sum() <= fails["B2"].sum()
        print(f"[T2-2] B1 < B0: {'PASS' if ok1 else 'FAIL'}   B3 <= B2: {'PASS' if ok2 else 'FAIL'}", flush=True)
    else:
        print("[T2-2] done", flush=True)

    out = full if ONLY == ALL else Path(f"results/t2_2_{TAG}_{'_'.join(ONLY)}.npz")
    np.savez(out, shots=SHOTS, p=P, snr=SNR, beta=BETA, p_leak=P_LEAK, p_seep=P_SEEP,
             rho=rho, p_s_avg=p_s_avg,
             **{f"fail_{k}": v for k, v in fails.items()}, **{f"sec_{k}": v for k, v in secs.items()},
             flags_b2=flags_b2, flags_b3=flags_b3, leaked_aux=s.leaked_aux, post_b4=post_b4,
             b2_precision=st2.precision, b2_recall=st2.recall, b3_precision=st3.precision, b3_recall=st3.recall)


if __name__ == "__main__":
    main()
