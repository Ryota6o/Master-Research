"""T3-3 学習スクリプト。

    nohup python -u -m leakdec.learn.train --repr raw --out runs/r1_raw > runs/r1_raw.log 2>&1 &

L = BCE(論理 12) + λ BCE(リーク, トークンごと)。損失履歴は npz に保存（案A の教訓）。
停滞相（論理 BCE ≈ 0.65 に約 2 万ステップ）で見切らない（RULES R2-3）。
検証は 2 組：固定 20,000 件（val_seed）と、T2-2 と同一の seed=0 の 2,000 件
（B1 0.159 / ORC 0.106 と同じショット。fail_mask を保存して対応のある比較に使う）。
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from leakdec.learn.dataset import DataConfig, make_loader, sample_tokens, token_spec
from leakdec.learn.model import SyndromeTransformer, logical_failures, losses

T2_2_SEED, T2_2_SHOTS = 0, 2000


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", choices=["raw", "soft", "hard"], required=True)
    ap.add_argument("--single-shot", action="store_true")
    ap.add_argument("--code", default="bbc-72-12-6")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--snr", type=float, default=8.4)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--p-leak", type=float, default=0.005)
    # AUDIT(2026-09-16) 【設計・重要】持続長掃引（T4-2）を p_leak 固定で p_seep だけ振ると定常リーク率 ρ が一緒に動く：
    #   p_seep=1 → ρ=0.50%、0.2 → 2.45%、0.1 → 4.8%、0.0333 → 13.1%（p_leak=0.02 の 9% で符号が崩壊した実績あり）。
    #   持続性の効果とリーク率の効果が混ざるので、ρ を固定して p_leak = ρ·p_seep/(1-ρ+ρ·p_seep) と解く。
    #   例 ρ=0.0245：p_seep=1 → p_leak=0.0245、0.5 → 0.0124、0.2 → 0.0050、0.1 → 0.00251、0.0333 → 0.00084。
    #   また T=8 ラウンドなので持続長 30 は「ショット全体でリーク」と同じ。掃引の上端は 1/p_seep ≈ 10 程度で十分。
    # FIX:
    # ap.add_argument("--rho", type=float, default=None, help="指定時は定常リーク率を固定し p_leak を p_seep から解く")
    # if args.rho is not None:
    #     args.p_leak = args.rho * args.p_seep / (1 - args.rho + args.rho * args.p_seep)
    ap.add_argument("--p-seep", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-shots", type=int, default=20_000)
    ap.add_argument("--val-seed", type=int, default=12345)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate(model, batch, device, chunk=1000):
    model.eval()
    fails, leak_p = [], []
    l_log = l_leak = 0.0
    for i in range(0, len(batch), chunk):
        x = batch.x[i:i + chunk].to(device)
        y = batch.logical[i:i + chunk].to(device)
        z = batch.leak[i:i + chunk].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logical, leak = model(x)
        logical, leak = logical.float(), leak.float()
        _, a, b = losses(logical, leak, y, z)
        l_log += a.item() * len(x)
        l_leak += b.item() * len(x)
        fails.append(logical_failures(logical, y).cpu())
        leak_p.append(torch.sigmoid(leak).cpu())
    model.train()
    fails = torch.cat(fails).numpy()
    leak_p = torch.cat(leak_p).numpy()
    truth = batch.leak.numpy() > 0.5
    flag = leak_p > 0.5
    tp = (flag & truth).sum()
    prec = tp / max(1, flag.sum())
    rec = tp / max(1, truth.sum())
    return dict(ler=float(fails.mean()), fails=int(fails.sum()), n=len(fails),
                logical_bce=l_log / len(fails), leak_bce=l_leak / len(fails),
                leak_precision=float(prec), leak_recall=float(rec)), fails, leak_p


def main():
    args = parse()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg = DataConfig(code_name=args.code, num_rounds=args.rounds, p=args.p, snr=args.snr, beta=args.beta,
                     p_leak=args.p_leak, p_seep=args.p_seep)
    sampler = cfg.make_sampler()
    spec = token_spec(args.repr, sampler.cmap)
    model = SyndromeTransformer(spec.d_in, spec.qubit_idx, spec.round_idx,
                                num_qubits=sampler.cmap.num_aux_qubits, num_rounds=sampler.cmap.num_aux_rounds,
                                d=args.d, layers=args.layers, heads=args.heads, ff=args.ff,
                                single_shot=args.single_shot).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))

    val, _ = sample_tokens(sampler, args.repr, args.val_shots, seed=args.val_seed)
    t22, _ = sample_tokens(sampler, args.repr, T2_2_SHOTS, seed=T2_2_SEED)

    hist = {k: [] for k in ("step", "loss", "logical", "leak", "lr", "train_ler")}
    evals = []
    start = 0
    ckpt_path = args.out / "last.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"] + 1
        hist = ck["hist"]
        evals = ck["evals"]
        print(f"[resume] from step {start}", flush=True)
    else:
        (args.out / "config.json").write_text(json.dumps({**vars(args), "out": str(args.out),
                                                           "n_params": n_params, "num_tokens": spec.num_tokens}, indent=2))
    print(f"[config] repr={args.repr} single_shot={args.single_shot} tokens={spec.num_tokens} params={n_params:,} "
          f"p={args.p} snr={args.snr} beta={args.beta} p_leak={args.p_leak} p_seep={args.p_seep} "
          f"steps={args.steps} batch={args.batch} device={device}", flush=True)

    loader = make_loader(cfg, args.repr, args.batch, num_workers=args.workers, base_seed=args.seed)
    it = iter(loader)
    # AUDIT(2026-09-16) 【最適化】実測 3〜4 it/s（4 本同時 + sweep_p 12 ワーカで CPU 24 本が飽和、主プロセスも CPU 100%）。
    #   d=128 のカーネルは小さく、カーネル起動律速になっている疑いが強い（GPU 使用率 100% は「何か動いている」だけ）。
    #   200k step ≈ 17 時間/本。候補：torch.compile（起動回数を減らす）、同時起動本数を減らす、sweep_p を学習と重ねない。
    # FIX（効果は要計測。bool マスク付きの r4 はコンパイルで落ちる可能性があるので単発は除外）:
    # if device.type == "cuda" and not args.single_shot:
    #     model = torch.compile(model)
    model.train()
    t0 = time.perf_counter()
    # AUDIT(2026-09-16) 【バグ・軽微】--resume で best が inf に戻るので、再開直後の評価が以前より悪くても best.pt を上書きする。
    # FIX: best = min((e["val"]["ler"] for e in evals), default=float("inf"))
    best = float("inf")
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        b = next(it).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logical, leak = model(b.x)
        loss, l1, l2 = losses(logical.float(), leak.float(), b.logical, b.leak, args.lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            tl = logical_failures(logical.float(), b.logical).float().mean().item()
            for k, v in (("step", step), ("loss", loss.item()), ("logical", l1.item()), ("leak", l2.item()),
                         ("lr", lr_at(step, args)), ("train_ler", tl)):
                hist[k].append(v)
            # AUDIT(2026-09-16) 【表示バグ】評価直後は t0 がリセットされるので、次のログの it/s は 1 ステップ分の時間から
            #   計算され 330 it/s のような値が出る（r1 の step 5000）。history.npz には影響しない。
            # FIX: 前回ログのステップ番号を持ち、経過ステップ数で割る
            # steps_done = step - last_log_step; last_log_step = step   # ループ前に last_log_step = start
            # ... f"{steps_done / dt:.1f} it/s"
            dt = time.perf_counter() - t0
            print(f"[{step:7d}] loss={loss.item():.4f} logical={l1.item():.4f} leak={l2.item():.4f} "
                  f"train_ler={tl:.3f} lr={lr_at(step, args):.2e} {args.log_every / dt:.1f} it/s", flush=True)
            t0 = time.perf_counter()
            np.savez(args.out / "history.npz", **{k: np.asarray(v) for k, v in hist.items()})

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev_val, _, _ = evaluate(model, val, device)
            ev_t22, fails_t22, leak_t22 = evaluate(model, t22, device)
            evals.append(dict(step=step + 1, val=ev_val, t22=ev_t22))
            print(f"[eval {step + 1}] val LER={ev_val['ler']:.4f} ({ev_val['fails']}/{ev_val['n']}) "
                  f"leak P/R={ev_val['leak_precision']:.3f}/{ev_val['leak_recall']:.3f} | "
                  f"T2-2 shots LER={ev_t22['ler']:.4f} ({ev_t22['fails']}/{ev_t22['n']})  "
                  f"[B1 0.159, ORC 0.106]", flush=True)
            (args.out / "evals.json").write_text(json.dumps(evals, indent=1))
            ck = dict(model=model.state_dict(), opt=opt.state_dict(), step=step, hist=hist, evals=evals,
                      args=vars(args) | {"out": str(args.out)})
            torch.save(ck, ckpt_path)
            if ev_val["ler"] < best:
                best = ev_val["ler"]
                torch.save(ck, args.out / "best.pt")
                np.savez(args.out / "t22_best.npz", fail_mask=fails_t22, leak_prob=leak_t22, step=step + 1)
            t0 = time.perf_counter()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
