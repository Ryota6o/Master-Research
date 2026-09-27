"""T2-1 受け入れ基準：既存 eval_postselect_baselines.py と同一ショットで LER が一致するか。

    nohup python -u scripts/run_t2_1_check.py > results/t2_1_check.log 2>&1 &

既存結果 ../myproject_copy/saved_results/postselect_baselines_N72_p0.004_n2000.npz
（serial, max_iter=200, osd_order=7, LER 0.0205）のショットごとの予測 L0 と突き合わせる。
(a) 拡張前の H、(b) 境界 144 列を追加した H（追加列の事前確率 ≈ 0）の両方で復号する。
"""
import sys
import time
from pathlib import Path

import numpy as np

from leakdec.decode.bposd_baseline import SoftDecodingProblem, build_soft_decoding_problem, decode_shots
from leakdec.sim.circuit_map import CircuitMap, default_map_path
from leakdec.sim.shots import load_or_make_shots, make_gym

P, N_SHOTS = 0.004, 2000
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 22
REF = Path("../myproject_copy/saved_results/postselect_baselines_N72_p0.004_n2000.npz")


def main():
    Path("results").mkdir(exist_ok=True)
    gym = make_gym(72, P)
    syndro, logerr = load_or_make_shots(gym, 72, P, N_SHOTS)
    ref = np.load(REF)
    L0_ref, wrong_ref = ref["L0"].astype(bool), ref["wrong_base"].astype(bool)
    assert ref["bp_schedule"] == "serial" and int(ref["max_iter"]) == 200 and int(ref["osd_order"]) == 7
    assert np.array_equal(ref["logerr"].astype(bool), logerr.astype(bool)), "ショット集合が違う"
    print(f"[ref] LER = {wrong_ref.sum()}/{N_SHOTS} = {wrong_ref.mean():.4f}", flush=True)

    H, L, pri = (gym.get_spacetime_parity_check_matrix(), gym.get_logical_decoding_matrix(),
                 gym.get_channel_probabalities())
    cmap = CircuitMap.load(default_map_path("bbc-72-12-6", "X", 6))
    plain = SoftDecodingProblem(H.astype(bool), L.astype(bool), pri.astype(np.float64),
                                np.zeros(0, dtype=np.int64), 0)
    aug = build_soft_decoding_problem(H, L, pri, cmap)

    out = {}
    for name, prob in (("plain", plain), ("augmented", aug)):
        t = time.time()
        pred, conv, _ = decode_shots(prob, syndro.astype(bool), None, n_workers=WORKERS)
        wrong = (pred != logerr.astype(bool)).any(axis=1)
        same_pred = (pred == L0_ref).all(axis=1)
        print(f"[{name}] H={prob.H.shape} {time.time() - t:.0f}s  BP conv={conv.mean():.3f}  "
              f"LER = {wrong.sum()}/{N_SHOTS} = {wrong.mean():.4f}  "
              f"予測が既存と一致したショット = {same_pred.sum()}/{N_SHOTS}  "
              f"正誤一致 = {(wrong == wrong_ref).sum()}/{N_SHOTS}", flush=True)
        out[name] = dict(pred=pred, wrong=wrong, conv=conv)

    verdict = "PASS" if all(v["wrong"].sum() == wrong_ref.sum() for v in out.values()) else "CHECK"
    print(f"[T2-1] LER 一致: {verdict}", flush=True)
    np.savez("results/t2_1_check.npz", wrong_ref=wrong_ref, L0_ref=L0_ref,
             **{f"{k}_{f}": v[f] for k, v in out.items() for f in ("pred", "wrong", "conv")})


if __name__ == "__main__":
    main()
