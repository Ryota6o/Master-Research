"""T1-6 関門の 3 番目：同一ショットで B0 (hard) と B1 (soft) の BP-OSD を比較する。

    nohup python -u scripts/run_gate_t1_6.py > results/gate_t1_6.log 2>&1 &

動作点：p=0.004、SNR=8.4（β=0 なら p_S≈0.02≈5p。Riverlane が利得の大きい領域と述べる p_S≫p 側）、
β=0.05、リーク無し。B0 の事前確率も平均 p_S で更新する（hard を弱くしない。RULES R1）。
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from leakdec.decode.bposd_baseline import (build_soft_decoding_problem, classification_error_from_llr,
                            decode_shots, update_priors, with_static_priors)
from leakdec.sim.circuit_map import CircuitMap, default_map_path
from leakdec.sim.gym import QECGym
from leakdec.sim.iq_model import IQModel
from leakdec.sim.soft_syndrome import SoftSyndromeSampler

P = 0.004
SNR, BETA = 8.4, 0.05
SHOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 22
SEED = 0


def cp_upper(k, n, alpha=0.05):
    return 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)


def cp_lower(k, n, alpha=0.05):
    return 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)


def report(name, fails, n):
    print(f"  {name}: {fails}/{n} = {fails / n:.4f}  "
          f"[CP95%: {cp_lower(fails, n):.4f}, {cp_upper(fails, n):.4f}]", flush=True)


def main():
    Path("results").mkdir(exist_ok=True)
    gym = QECGym("bbc-72-12-6", "X", "circuit", P, num_rounds=6,
                 measure_both=True, load_saved_logical_ops=True)
    cmap = CircuitMap.load(default_map_path("bbc-72-12-6", "X", 6))
    iq = IQModel(snr=SNR, beta=BETA)
    p_s_avg = iq.readout_error_probability()
    print(f"[config] p={P} snr={SNR} beta={BETA} p_S_avg={p_s_avg:.4f} shots={SHOTS} workers={WORKERS}", flush=True)

    sampler = SoftSyndromeSampler(gym._circuit, iq, None, cmap)
    s = sampler.sample(SHOTS, seed=SEED)
    print(f"[data] detector rate ideal={s.detectors_ideal.mean():.4f} hard={s.detectors_hard.mean():.4f}  "
          f"obs flip rate={s.observables.any(axis=1).mean():.4f}", flush=True)

    problem = build_soft_decoding_problem(
        gym.get_spacetime_parity_check_matrix(), gym.get_logical_decoding_matrix(),
        gym.get_channel_probabalities(), cmap)
    print(f"[problem] H={problem.H.shape} added={problem.num_added}", flush=True)

    truth = s.observables
    results = {}

    # 参考：IQ 雑音なし（既存パイプライン相当）
    t = time.time()
    pred, conv, _ = decode_shots(problem, s.detectors_ideal, None, n_workers=WORKERS)
    fail_ideal = (pred != truth).any(axis=1)
    print(f"[ref ] no-IQ-noise: {time.time() - t:.0f}s  BP conv={conv.mean():.3f}", flush=True)
    report("LER(ref, no IQ noise)", int(fail_ideal.sum()), SHOTS)

    # B0：hard 検出器 + 平均 p_S で更新した静的事前確率
    t = time.time()
    prob_b0 = with_static_priors(problem, p_s_avg)
    pred, conv, _ = decode_shots(prob_b0, s.detectors_hard, None, n_workers=WORKERS)
    fail_b0 = (pred != truth).any(axis=1)
    print(f"[B0  ] hard: {time.time() - t:.0f}s  BP conv={conv.mean():.3f}", flush=True)
    report("LER(B0 hard)", int(fail_b0.sum()), SHOTS)

    # B1：hard 検出器 + ショットごとの p_S で更新
    t = time.time()
    p_s = classification_error_from_llr(s.meas_llr)
    priors = update_priors(problem, p_s)
    pred, conv, _ = decode_shots(problem, s.detectors_hard, priors, n_workers=WORKERS)
    fail_b1 = (pred != truth).any(axis=1)
    print(f"[B1  ] soft: {time.time() - t:.0f}s  BP conv={conv.mean():.3f}", flush=True)
    report("LER(B1 soft)", int(fail_b1.sum()), SHOTS)

    only_b0 = int((fail_b0 & ~fail_b1).sum())
    only_b1 = int((fail_b1 & ~fail_b0).sum())
    print(f"[pair] B0 だけ失敗={only_b0}  B1 だけ失敗={only_b1}  両方失敗={int((fail_b0 & fail_b1).sum())}", flush=True)
    verdict = "PASS" if fail_b1.sum() < fail_b0.sum() else "FAIL"
    print(f"[gate] B1 < B0 : {verdict}", flush=True)

    np.savez("results/gate_t1_6.npz", shots=SHOTS, p=P, snr=SNR, beta=BETA, p_s_avg=p_s_avg,
             fail_ideal=int(fail_ideal.sum()), fail_b0=int(fail_b0.sum()), fail_b1=int(fail_b1.sum()),
             only_b0_fails=only_b0, only_b1_fails=only_b1,
             fail_ideal_mask=fail_ideal, fail_b0_mask=fail_b0, fail_b1_mask=fail_b1)


if __name__ == "__main__":
    main()
