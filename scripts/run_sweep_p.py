"""T4-1：物理エラー率 p を振って B0 / B1 / ORC を評価し、ショット集合を固定して保存する。

    nohup python -u scripts/run_sweep_p.py --workers 12 > results/sweep_p/log 2>&1 &

各 p のショットは seed = 1000 + i で生成する（学習型の評価も同じ seed で同一ショットを再現する）。
出力：results/sweep_p/p{p}.npz（fail_*, sec_*, seed）。Λ は使わない。0 誤りは CP95% 上限を併記。
"""
import argparse
import time
from pathlib import Path

import numpy as np

from leakdec.decode.bposd_baseline import (build_soft_decoding_problem, classification_error_from_llr,
                            decode_shots, update_priors, with_static_priors)
from leakdec.learn.dataset import DataConfig
from leakdec.sim.gym import QECGym
from leakdec.decode.leak_baselines import p_s_with_leak_flags
from leakdec.analysis.stats import latency_line, ler_line

DEFAULT_PS = [0.001, 0.002, 0.003, 0.004, 0.006]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=float, nargs="+", default=DEFAULT_PS)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--snr", type=float, default=8.4)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--p-leak", type=float, default=0.005)
    ap.add_argument("--p-seep", type=float, default=0.2)
    ap.add_argument("--conditions", default="B0,B1,ORC")
    ap.add_argument("--out", type=Path, default=Path("results/sweep_p"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conds = args.conditions.split(",")

    # AUDIT(2026-09-16) 【設計】(1) SNR を固定して p を振ると p_S/p 比が 32（p=0.001）→ 5（p=0.006）と動く。
    #   plan01 §6.2 は p_S/p 比を分離すべきと言っているので、「p_S = 比 × p を保つ」モードを持つべき。
    #   (2) 学習型は p=0.004 で学習したモデルを他の p のショットに流用することになり、分布シフトが入る。
    #   古典側は DEM の事前確率が p ごとに更新されるので不公平。学習型の掃引には p を範囲からサンプルして学習するか、
    #   p ごとに学習し直す必要がある（train.py の --p を掃引、または DataConfig に p の範囲を持たせる）。
    #   (3) p=0.004 の seed は 1000+3 で、T2-2（seed=0）とは別ショット。同じ p で B1 が 0.159 と異なる値になり混乱する。
    # FIX（p_S/p 比固定モード）:
    # from scipy.optimize import brentq
    # def snr_for_p_s(target, beta):
    #     return brentq(lambda s: IQModel(snr=s, beta=beta).readout_error_probability() - target, 1.0, 200.0)
    # ap.add_argument("--ps-ratio", type=float, default=None, help="指定時 p_S = ratio·p となる SNR を p ごとに解く")
    # snr = args.snr if args.ps_ratio is None else snr_for_p_s(args.ps_ratio * p, args.beta)
    for i, p in enumerate(args.ps):
        seed = 1000 + i
        cfg = DataConfig(p=p, snr=args.snr, beta=args.beta, p_leak=args.p_leak, p_seep=args.p_seep)
        sampler = cfg.make_sampler()
        s = sampler.sample(args.shots, seed=seed)
        gym = QECGym(cfg.code_name, "X", "circuit", p, num_rounds=cfg.num_rounds,
                     measure_both=True, load_saved_logical_ops=True)
        problem = build_soft_decoding_problem(gym.get_spacetime_parity_check_matrix(),
                                              gym.get_logical_decoding_matrix(),
                                              gym.get_channel_probabalities(), sampler.cmap)
        p_s_avg = sampler.iq_model.readout_error_probability()
        print(f"[p={p}] seed={seed} det rate hard={s.detectors_hard.mean():.4f} "
              f"p_S_avg={p_s_avg:.4f} leaked={s.leaked_aux.mean():.4f}", flush=True)
        priors = {
            "B0": (with_static_priors(problem, p_s_avg), None),
            "B1": (problem, update_priors(problem, classification_error_from_llr(s.meas_llr))),
            "ORC": (problem, update_priors(problem, p_s_with_leak_flags(s.meas_llr, s.leaked_aux))),
        }
        out = dict(p=p, seed=seed, shots=args.shots, snr=args.snr, beta=args.beta,
                   p_leak=args.p_leak, p_seep=args.p_seep, p_s_avg=p_s_avg)
        for name in conds:
            prob, pri = priors[name]
            t = time.time()
            pred, conv, sec = decode_shots(prob, s.detectors_hard, pri, n_workers=args.workers)
            fails = (pred != s.observables).any(axis=1)
            out[f"fail_{name}"], out[f"sec_{name}"] = fails, sec
            print(f"  [{name:3s}] {time.time() - t:.0f}s  BP conv={conv.mean():.3f}  "
                  f"{ler_line('LER', int(fails.sum()), args.shots)}  latency {latency_line(sec)}", flush=True)
        np.savez(args.out / f"p{p}.npz", **out)
    print("[sweep_p] done", flush=True)


if __name__ == "__main__":
    main()
