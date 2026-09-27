"""学習済みモデルを T2-2 と同一ショット（seed=0, 2000）で評価し、B1/ORC と対応のある比較をする。

    python -m leakdec.analysis.evaluate runs/r1_raw_seq [--hybrid] [--workers 22]

  - LER（CP95%）と McNemar（B1 / B3s / ORC に対して）。T2-2 の npz が同じ動作点のときのみ
  - リークヘッドの precision / recall / AUC（対 真の ℓ）
  - --hybrid（raw のみ、D5）：リーク確率 p_ℓ を p_S = (1-p_ℓ)·p_S,2状態 + ½ p_ℓ に変換して
    BP-OSD に渡す。学習したリーク推論がオラクルの余地をどれだけ取るかを古典側で測る
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from leakdec.decode.bposd_baseline import (build_soft_decoding_problem, classification_error_from_llr,
                            decode_shots, update_priors)
from leakdec.learn.dataset import DataConfig, sample_tokens, token_spec
from leakdec.sim.gym import QECGym
from leakdec.decode.leak_baselines import FlagStats
from leakdec.decode.leak_hmm import hmm_leak_posterior
from leakdec.learn.model import SyndromeTransformer
from leakdec.analysis.stats import latency_line, ler_line, mcnemar_counts
from leakdec.learn.train import T2_2_SEED, T2_2_SHOTS, evaluate


@torch.no_grad()
def inference_latency(model, x: torch.Tensor, device, n: int) -> np.ndarray:
    """batch=1 の forward 1 回あたりの秒数 (n,)。GPU は synchronize で壁時計を測る。"""
    model = model.to(device).eval()
    xs = x[:n].to(device)
    for i in range(min(20, n)):
        model(xs[i:i + 1])
    times = np.empty(len(xs))
    for i in range(len(xs)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(xs[i:i + 1])
        if device.type == "cuda":
            torch.cuda.synchronize()
        times[i] = time.perf_counter() - t0
    return times


def auc(score: np.ndarray, truth: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos, n_neg = truth.sum(), (~truth).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--hybrid-shots", type=int, default=T2_2_SHOTS, help="動作確認用に先頭 N ショットだけ復号")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--latency-shots", type=int, default=2000, help="0 でレイテンシ計測を省略")
    ap.add_argument("--cpu-latency-shots", type=int, default=200)
    args = ap.parse_args()

    ck = torch.load(args.run / args.ckpt, map_location="cpu")
    a = ck["args"]
    cfg = DataConfig(code_name=a.get("code", "bbc-72-12-6"), num_rounds=a.get("rounds", 6),
                     p=a["p"], snr=a["snr"], beta=a["beta"], p_leak=a["p_leak"], p_seep=a["p_seep"])
    sampler = cfg.make_sampler()
    spec = token_spec(a["repr"], sampler.cmap)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyndromeTransformer(spec.d_in, spec.qubit_idx, spec.round_idx,
                                num_qubits=sampler.cmap.num_aux_qubits, num_rounds=sampler.cmap.num_aux_rounds,
                                d=a["d"], layers=a["layers"], heads=a["heads"], ff=a["ff"],
                                single_shot=a["single_shot"]).to(device)
    model.load_state_dict(ck["model"])

    batch, shots = sample_tokens(sampler, a["repr"], T2_2_SHOTS, seed=T2_2_SEED)
    ev, fails, leak_p = evaluate(model, batch, device)
    truth_leak = batch.leak.numpy() > 0.5
    print(f"[{args.run.name}] repr={a['repr']} single_shot={a['single_shot']} step={ck['step'] + 1}")
    print("  " + ler_line("LER(T2-2 shots)", ev["fails"], ev["n"]))
    st = FlagStats.of(leak_p > 0.5, truth_leak)
    print(f"  leak head: precision={st.precision:.3f} recall={st.recall:.3f} "
          f"AUC={auc(leak_p.ravel(), truth_leak.ravel()):.3f} flagged={st.flagged_rate:.4f}")
    b4_auc = None
    if a["repr"] == "raw" and sampler.leakage is not None:
        # 参照：B4（HMM）は量子ビットごとの時間方向推論の上限。符号構造とビットのラウンド間相関は使わないので
        # 学習型はこれを超えうる（step 10000 の raw で AUC 0.969 対 0.958）
        post_b4 = hmm_leak_posterior(shots.iq, sampler.iq_model, sampler.leakage, sampler.cmap)
        st4 = FlagStats.of(post_b4 > 0.5, truth_leak)
        b4_auc = auc(post_b4.ravel(), truth_leak.ravel())
        print(f"  B4 (HMM, 量子ビットごと時間方向の上限): precision={st4.precision:.3f} recall={st4.recall:.3f} AUC={b4_auc:.3f}")

    tag = f"pl{a['p_leak']}_ps{a['p_seep']}"
    ref = Path(f"results/t2_2_{tag}.npz")
    orc = Path(f"results/t2_2_{tag}_ORC.npz")
    # AUDIT(2026-09-16) 【バグ】符号・ラウンド数を見ていない。144 符号の run（--code bbc-144-12-12）も 2000 ショットなので、
    #   72 符号の T2-2 npz と形が合ってしまい、別ショットとの「対応のある比較」を黙って出力する。
    # FIX:
    # same_point = (ref.exists() and a["p"] == 0.004 and a["snr"] == 8.4 and a["beta"] == 0.05
    #               and a.get("code", "bbc-72-12-6") == "bbc-72-12-6" and a.get("rounds", 6) == 6)
    same_point = ref.exists() and a["p"] == 0.004 and a["snr"] == 8.4 and a["beta"] == 0.05
    if same_point:
        r = np.load(ref)
        baselines = {k[5:]: r[k] for k in r.files if k.startswith("fail_")}
        if orc.exists():
            baselines["ORC"] = np.load(orc)["fail_ORC"]
        for name in ("B1", "B3s", "ORC"):
            if name in baselines:
                only_b, only_m = mcnemar_counts(baselines[name], fails)
                print(f"  vs {name} ({baselines[name].mean():.4f}): {name}だけ失敗={only_b} モデルだけ失敗={only_m}")
    else:
        print("  （T2-2 の結果と動作点が違うので対応のある比較は省略）")

    out = dict(run=str(args.run), step=ck["step"] + 1, **ev,
               leak_auc=auc(leak_p.ravel(), truth_leak.ravel()), b4_leak_auc=b4_auc)
    lat = {}
    if args.latency_shots > 0:
        # AUDIT(2026-09-16) 【計測の妥当性】(1) 学習 4 本 + sweep_p 12 ワーカが同時に走る状態で測ると GPU/CPU の
        #   競合で数倍遅く出る。レイテンシは他ジョブを止めた状態で別途測ること（T4-4 の数値に使う前に再計測）。
        #   (2) ここは autocast 無しの fp32 で測っている。学習・評価経路は bf16 なので、bf16 でも測って両方併記する。
        # FIX（bf16 でも測る）:
        # with torch.autocast("cuda", dtype=torch.bfloat16):
        #     t_gpu_bf16 = inference_latency(model, batch.x, device, args.latency_shots)
        # 軸3：誤りパターンに依らず固定時間。BP-OSD（中央値 6.3 s / p99 6.8 s）と同じ 2000 ショットで測る
        if device.type == "cuda":
            t_gpu = inference_latency(model, batch.x, device, args.latency_shots)
            lat["gpu"] = t_gpu
            print(f"  latency GPU batch=1: {latency_line(t_gpu)}  (n={len(t_gpu)})")
        cpu_model = copy.deepcopy(model).to("cpu").float()
        t_cpu = inference_latency(cpu_model, batch.x, torch.device("cpu"), args.cpu_latency_shots)
        lat["cpu"] = t_cpu
        print(f"  latency CPU batch=1 ({torch.get_num_threads()} threads): {latency_line(t_cpu)}  (n={len(t_cpu)})")
        for k, v in lat.items():
            out[f"latency_{k}_median_s"] = float(np.median(v))
            out[f"latency_{k}_p99_s"] = float(np.percentile(v, 99))
    np.savez(args.run / "eval_t22.npz", fail_mask=fails, leak_prob=leak_p,
             **{f"latency_{k}": v for k, v in lat.items()})

    if args.hybrid:
        if a["repr"] != "raw":
            raise SystemExit("--hybrid は raw のみ（リーク確率が測定トークンと 1 対 1）")
        gym = QECGym(cfg.code_name, "X", "circuit", cfg.p, num_rounds=cfg.num_rounds,
                     measure_both=True, load_saved_logical_ops=True)
        problem = build_soft_decoding_problem(gym.get_spacetime_parity_check_matrix(),
                                              gym.get_logical_decoding_matrix(),
                                              gym.get_channel_probabalities(), sampler.cmap)
        n = args.hybrid_shots
        p2 = classification_error_from_llr(shots.meas_llr[:n])
        p_s = (1.0 - leak_p[:n]) * p2 + 0.5 * leak_p[:n]
        pred, conv, sec = decode_shots(problem, shots.detectors_hard[:n], update_priors(problem, p_s),
                                       n_workers=args.workers)
        fails_h = (pred != shots.observables[:n]).any(axis=1)
        print("  " + ler_line("LER(hybrid: leak head → BP-OSD)", int(fails_h.sum()), len(fails_h))
              + f"  BP conv={conv.mean():.3f}  latency {latency_line(sec)}")
        if same_point:
            for name in ("B1", "ORC"):
                if name in baselines:
                    print(f"  hybrid vs {name} (先頭 {n} ショット): {mcnemar_counts(baselines[name][:n], fails_h)}")
        out["hybrid_fails"] = int(fails_h.sum())
        np.savez(args.run / "eval_t22_hybrid.npz", fail_mask=fails_h, p_s=p_s, seconds=sec)

    (args.run / "eval_t22.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
