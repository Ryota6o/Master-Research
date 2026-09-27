"""学習と比較の図を PNG に出す。

    python -m leakdec.analysis.plot_runs runs/r1_raw_seq runs/r2_hard_seq runs/r3_soft_seq runs/r4_raw_single [--out results/fig]

出力（無いデータは黙って飛ばす）：
  fig/loss_curves.png   history.npz：論理 BCE / リーク BCE / train LER を run ごとに重ねる（移動平均）。
                        2 万 step の停滞相の目安線（RULES R2-3）とリーク教師の事前エントロピー線を入れる
  fig/eval_curves.png   evals.json：val LER と T2-2 ショット LER の推移に B1 / ORC の水平線。リーク P/R も
  fig/pl_vs_p.png       results/sweep_p/p*.npz：P_L vs p（B0/B1/ORC、CP95% 誤差棒、対数軸）。
                        runs/<run>/eval_sweep_p.json（未実装の学習型掃引評価）があれば同じ軸に重ねる
  fig/latency.png       results/t2_2_*.npz の sec_* と runs/<run>/eval_t22.npz の latency_* を対数軸の箱ひげで並べる
学習中でも history.npz / evals.json は随時更新されるので、途中経過の確認にそのまま使える。
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from leakdec.analysis.stats import cp_lower, cp_upper  # noqa: E402

PLATEAU_STEP = 20_000
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_run(run: Path) -> dict:
    out = dict(name=run.name, cfg=json.loads((run / "config.json").read_text()))
    if (run / "history.npz").exists():
        h = np.load(run / "history.npz")
        out["hist"] = {k: h[k] for k in h.files}
    if (run / "evals.json").exists():
        out["evals"] = json.loads((run / "evals.json").read_text())
    if (run / "eval_t22.npz").exists():
        e = np.load(run / "eval_t22.npz")
        out["latency"] = {k[8:]: e[k] for k in e.files if k.startswith("latency_")}
    if (run / "eval_sweep_p.json").exists():
        out["sweep"] = json.loads((run / "eval_sweep_p.json").read_text())
    return out


def label_of(r: dict) -> str:
    c = r["cfg"]
    return f"{r['name']} ({c['repr']}, {'単発' if c['single_shot'] else '系列'})"


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def leak_prior_entropy(cfg: dict) -> float:
    """リーク教師の事前エントロピー（学習しないときのリーク BCE）。検出器トークンは 2 測定の OR。"""
    pl, ps = cfg["p_leak"], cfg["p_seep"]
    if pl == 0:
        return 0.0
    rho = pl / (pl + ps * (1 - pl))
    if cfg["repr"] != "raw":
        stay = 1 - ps * (1 - pl)
        rho = 2 * rho - rho * stay  # P(ℓ_r ∪ ℓ_{r+1})
    return float(-(rho * np.log(rho) + (1 - rho) * np.log(1 - rho)))


# ---- 1. 損失曲線 ------------------------------------------------------------------------

def plot_losses(runs, out: Path, k: int):
    runs = [r for r in runs if "hist" in r]
    if not runs:
        return
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for i, r in enumerate(runs):
        h, c = r["hist"], COLORS[i % len(COLORS)]
        s = h["step"]
        for ax, key in zip(axes, ("logical", "leak", "train_ler")):
            y = smooth(h[key], k)
            ax.plot(s[len(s) - len(y):], y, color=c, label=label_of(r), lw=1.3)
        axes[1].axhline(leak_prior_entropy(r["cfg"]), color=c, ls=":", lw=0.8)
    axes[0].axhline(np.log(2), color="gray", ls="--", lw=0.8)
    axes[0].text(0, np.log(2), " ln 2（当てずっぽう）", va="bottom", fontsize=8, color="gray")
    axes[0].set_ylabel("論理 BCE（12 ビット平均）")
    axes[1].set_ylabel("リーク BCE（トークン平均）")
    axes[1].text(0, 0, " 点線 = 教師の事前エントロピー（学習しないときの値）", fontsize=8, color="gray", va="bottom")
    axes[2].set_ylabel("train LER（バッチ内）")
    axes[2].set_xlabel("step")
    for ax in axes:
        ax.axvline(PLATEAU_STEP, color="red", ls="--", lw=0.8)
        ax.grid(alpha=0.3)
    axes[0].text(PLATEAU_STEP, axes[0].get_ylim()[1], "2 万 step（ここまでは見切らない） ", color="red", fontsize=8, va="top", ha="right")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"学習曲線（移動平均 {k} 点 = {k * 100} step）")
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=130)
    plt.close(fig)
    print(f"[fig] {out / 'loss_curves.png'}")


# ---- 2. 評価の推移 ------------------------------------------------------------------------

def baseline_lers(cfg: dict) -> dict:
    tag = f"pl{cfg['p_leak']}_ps{cfg['p_seep']}"
    out = {}
    if not (cfg["p"] == 0.004 and cfg["snr"] == 8.4 and cfg["beta"] == 0.05
            and cfg.get("code", "bbc-72-12-6") == "bbc-72-12-6"):
        return out
    f = Path(f"results/t2_2_{tag}.npz")
    if f.exists():
        r = np.load(f)
        out.update({k[5:]: float(r[k].mean()) for k in r.files if k.startswith("fail_")})
    g = Path(f"results/t2_2_{tag}_ORC.npz")
    if g.exists():
        out["ORC"] = float(np.load(g)["fail_ORC"].mean())
    return out


def plot_evals(runs, out: Path):
    runs = [r for r in runs if r.get("evals")]
    if not runs:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for i, r in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        st = [e["step"] for e in r["evals"]]
        axes[0].plot(st, [e["val"]["ler"] for e in r["evals"]], "o-", color=c, label=label_of(r) + " val 2万")
        axes[0].plot(st, [e["t22"]["ler"] for e in r["evals"]], "s--", color=c, alpha=0.6, label=label_of(r) + " T2-2 2千")
        axes[1].plot(st, [e["val"]["leak_precision"] for e in r["evals"]], "o-", color=c, label=label_of(r) + " 適合率")
        axes[1].plot(st, [e["val"]["leak_recall"] for e in r["evals"]], "s--", color=c, alpha=0.6, label=label_of(r) + " 再現率")
    for name, ls in (("B1", "--"), ("B3s", ":"), ("ORC", "-.")):
        v = baseline_lers(runs[0]["cfg"]).get(name)
        if v is not None:
            axes[0].axhline(v, color="black", ls=ls, lw=0.9, label=f"{name}（BP-OSD, 同一 T2-2 ショット）{v:.3f}")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("LER")
    axes[0].axvline(PLATEAU_STEP, color="red", ls="--", lw=0.8)
    axes[1].set_ylabel("リークヘッド（閾値 0.5）")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("step")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("評価の推移（5000 step ごと）。soft/hard のリーク教師は検出器 2 測定の OR で raw と定義が違う")
    fig.tight_layout()
    fig.savefig(out / "eval_curves.png", dpi=130)
    plt.close(fig)
    print(f"[fig] {out / 'eval_curves.png'}")


# ---- 3. P_L vs p ---------------------------------------------------------------------------

def plot_pl_vs_p(runs, out: Path, sweep_dir: Path):
    files = sorted(sweep_dir.glob("p*.npz"), key=lambda f: float(f.stem[1:]))
    if not files and not any("sweep" in r for r in runs):
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    series: dict[str, list] = {}
    meta = None
    for f in files:
        d = np.load(f)
        meta = meta or dict(snr=float(d["snr"]), beta=float(d["beta"]), p_leak=float(d["p_leak"]),
                            p_seep=float(d["p_seep"]), p_s_avg=float(d["p_s_avg"]))
        for k in d.files:
            if k.startswith("fail_"):
                fails, n = int(d[k].sum()), int(d["shots"])
                series.setdefault(k[5:], []).append((float(d["p"]), fails, n))
    markers = {"B0": "s", "B1": "o", "ORC": "^"}
    for name, pts in series.items():
        pts.sort()
        p = np.array([q[0] for q in pts])
        k = np.array([q[1] for q in pts])
        n = np.array([q[2] for q in pts])
        ler = k / n
        lo = ler - np.array([cp_lower(a, b) for a, b in zip(k, n)])
        hi = np.array([cp_upper(a, b) for a, b in zip(k, n)]) - ler
        # 0 誤りは CP95% 上限だけを下向き矢印で示す（R3-4）
        zero = k == 0
        ax.errorbar(p[~zero], ler[~zero], yerr=[lo[~zero], hi[~zero]], fmt=markers.get(name, "x") + "-",
                    capsize=3, label=f"{name}（BP-OSD）", color="black" if name == "ORC" else None,
                    alpha=0.8 if name == "ORC" else 1.0)
        if zero.any():
            ax.errorbar(p[zero], ler[zero] + hi[zero], yerr=hi[zero] * 0.5, uplims=True, fmt="none", color="gray")
    for i, r in enumerate(runs):
        if "sweep" not in r:
            continue
        pts = sorted((float(q["p"]), int(q["fails"]), int(q["n"])) for q in r["sweep"])
        p = np.array([q[0] for q in pts]); k = np.array([q[1] for q in pts]); n = np.array([q[2] for q in pts])
        ler = k / n
        ax.errorbar(p, np.maximum(ler, 1e-6),
                    yerr=[ler - np.array([cp_lower(a, b) for a, b in zip(k, n)]),
                          np.array([cp_upper(a, b) for a, b in zip(k, n)]) - ler],
                    fmt="D-", capsize=3, color=COLORS[i % len(COLORS)], label=label_of(r))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("物理エラー率 p")
    ax.set_ylabel("論理誤り率 P_L（誤差棒 = CP95%）")
    if meta:
        ax.set_title(f"[[72,12,6]] X 基底 6 ラウンド, SNR={meta['snr']} β={meta['beta']} (p_S≈{meta['p_s_avg']:.3f} 固定), "
                     f"p_leak={meta['p_leak']} p_seep={meta['p_seep']}", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pl_vs_p.png", dpi=130)
    plt.close(fig)
    print(f"[fig] {out / 'pl_vs_p.png'}")


# ---- 4. レイテンシ ---------------------------------------------------------------------------

def plot_latency(runs, out: Path):
    data, labels = [], []
    for f in sorted(Path("results").glob("t2_2_*.npz")):
        d = np.load(f)
        for k in d.files:
            if k.startswith("sec_"):
                data.append(d[k]); labels.append(f"{k[4:]} BP-OSD\n(22 並列時)")
    for r in runs:
        for dev, sec in r.get("latency", {}).items():
            data.append(sec); labels.append(f"{r['name']}\n{dev} batch=1")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(data)), 5))
    ax.boxplot(data, tick_labels=labels, whis=(1, 99), showfliers=False)
    for i, s in enumerate(data, 1):
        ax.plot(i, np.percentile(s, 99), "rv", ms=5)
        ax.text(i, np.percentile(s, 99), f" p99 {np.percentile(s, 99):.3g}s\n med {np.median(s):.3g}s", fontsize=7, va="bottom")
    ax.set_yscale("log")
    ax.set_ylabel("1 ショットの復号時間 [s]（箱 = 四分位、ひげ = 1〜99%）")
    ax.set_title("レイテンシ分布。BP-OSD は 22 ワーカ同時実行下の壁時計（単独計測より遅め）", fontsize=9)
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=130)
    plt.close(fig)
    print(f"[fig] {out / 'latency.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/fig"))
    ap.add_argument("--smooth", type=int, default=5, help="損失曲線の移動平均の点数（1 点 = log_every step）")
    ap.add_argument("--sweep-dir", type=Path, default=Path("results/sweep_p"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]
    runs = [load_run(r) for r in args.runs if (r / "config.json").exists()]
    plot_losses(runs, args.out, args.smooth)
    plot_evals(runs, args.out)
    plot_pl_vs_p(runs, args.out, args.sweep_dir)
    plot_latency(runs, args.out)


if __name__ == "__main__":
    main()
