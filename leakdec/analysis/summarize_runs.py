"""学習の結果を 1 つの表にまとめる（T4-3 アブレーション、T4-2 持続長掃引）。

    python -m leakdec.analysis.summarize_runs runs/r1_raw_seq runs/r2_hard_seq runs/r3_soft_seq runs/r4_raw_single

各 run の eval_t22.json（evaluate.py の出力）を優先し、無ければ evals.json の最新を使う。
同じ動作点の T2-2 結果があれば B1 / ORC との McNemar を再計算する。Markdown を標準出力と
results/summary_runs.md に出す。
"""
import json
import sys
from pathlib import Path

import numpy as np

from leakdec.analysis.stats import cp_lower, cp_upper, mcnemar_counts


def load_run(run: Path) -> dict:
    cfg = json.loads((run / "config.json").read_text())
    # AUDIT(2026-09-16) 表の「リーク P/R/AUC」は raw（測定トークン、正例率 2.4%）と soft/hard（検出器トークン、
    #   教師は 2 測定の OR、正例率 2.9%）で教師の定義が違うので横並びに比べられない。列名に注記を付けるか、
    #   soft/hard は検出器の OR 教師である旨を脚注にする。
    # FIX: row に code を持たせる（baselines_for で使う）
    # row = dict(run=run.name, code=cfg.get("code", "bbc-72-12-6"), repr=cfg["repr"], ...
    row = dict(run=run.name, repr=cfg["repr"], single_shot=cfg["single_shot"],
               p_leak=cfg["p_leak"], p_seep=cfg["p_seep"], p=cfg["p"], snr=cfg["snr"], beta=cfg["beta"])
    ev_path = run / "eval_t22.json"
    if ev_path.exists():
        ev = json.loads(ev_path.read_text())
        row.update(step=ev["step"], fails=ev["fails"], n=ev["n"], leak_precision=ev["leak_precision"],
                   leak_recall=ev["leak_recall"], leak_auc=ev.get("leak_auc"),
                   hybrid_fails=ev.get("hybrid_fails"),
                   lat_gpu=ev.get("latency_gpu_median_s"), lat_gpu_p99=ev.get("latency_gpu_p99_s"),
                   lat_cpu=ev.get("latency_cpu_median_s"), lat_cpu_p99=ev.get("latency_cpu_p99_s"),
                   source="eval_t22")
        mask_path = run / "eval_t22.npz"
        row["fail_mask"] = np.load(mask_path)["fail_mask"] if mask_path.exists() else None
    elif (run / "evals.json").exists():
        last = json.loads((run / "evals.json").read_text())[-1]
        t = last["t22"]
        row.update(step=last["step"], fails=t["fails"], n=t["n"], leak_precision=t["leak_precision"],
                   leak_recall=t["leak_recall"], leak_auc=None, hybrid_fails=None,
                   lat_gpu=None, lat_gpu_p99=None, lat_cpu=None, lat_cpu_p99=None, source="evals(last)")
        best = run / "t22_best.npz"
        row["fail_mask"] = np.load(best)["fail_mask"] if best.exists() else None
    else:
        row.update(step=None, fails=None, n=None, source="(未評価)", fail_mask=None)
    return row


def baselines_for(row: dict) -> dict:
    tag = f"pl{row['p_leak']}_ps{row['p_seep']}"
    out = {}
    # AUDIT(2026-09-16) 【バグ】evaluate.py と同じ：符号を見ていないので 144 符号の run が 72 符号の B1/ORC と対にされる。
    # FIX:
    # if (row["p"] == 0.004 and row["snr"] == 8.4 and row["beta"] == 0.05
    #         and row.get("code", "bbc-72-12-6") == "bbc-72-12-6"):
    if row["p"] == 0.004 and row["snr"] == 8.4 and row["beta"] == 0.05:
        f = Path(f"results/t2_2_{tag}.npz")
        if f.exists():
            r = np.load(f)
            out.update({k[5:]: r[k] for k in r.files if k.startswith("fail_")})
        g = Path(f"results/t2_2_{tag}_ORC.npz")
        if g.exists():
            out["ORC"] = np.load(g)["fail_ORC"]
    return out


def fmt_ler(k, n):
    if k is None:
        return "—"
    return f"{k / n:.4f} [{cp_lower(k, n):.4f}, {cp_upper(k, n):.4f}]"


def main(paths):
    rows = [load_run(Path(p)) for p in paths]
    lines = ["| run | 表現 | 時間方向 | 1/p_seep | step | LER (CP95%) | vs B1 (B1だけ, モデルだけ) | vs ORC | リーク P/R/AUC | hybrid LER | 推論 GPU/CPU 中央値 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b = baselines_for(r)
        vs_b1 = vs_orc = "—"
        if r.get("fail_mask") is not None:
            if "B1" in b:
                vs_b1 = f"{b['B1'].mean():.4f} ({mcnemar_counts(b['B1'], r['fail_mask'])})"
            if "ORC" in b:
                vs_orc = f"{b['ORC'].mean():.4f} ({mcnemar_counts(b['ORC'], r['fail_mask'])})"
        pr = "—" if r.get("leak_precision") is None else (
            f"{r['leak_precision']:.2f}/{r['leak_recall']:.2f}/" + ("—" if r.get("leak_auc") is None else f"{r['leak_auc']:.3f}"))
        hyb = "—" if r.get("hybrid_fails") is None else fmt_ler(r["hybrid_fails"], r["n"])
        lat = "—" if r.get("lat_gpu") is None else (
            f"{r['lat_gpu'] * 1e3:.2f} ms / " + ("—" if r.get("lat_cpu") is None else f"{r['lat_cpu'] * 1e3:.1f} ms"))
        lines.append(f"| {r['run']} | {r['repr']} | {'なし' if r['single_shot'] else 'あり'} | "
                     f"{1 / r['p_seep']:.0f} | {r.get('step')} ({r['source']}) | {fmt_ler(r.get('fails'), r.get('n'))} | "
                     f"{vs_b1} | {vs_orc} | {pr} | {hyb} | {lat} |")
    text = "\n".join(lines)
    print(text)
    Path("results").mkdir(exist_ok=True)
    Path("results/summary_runs.md").write_text(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
