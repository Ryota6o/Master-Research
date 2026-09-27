"""soft 測定 (LLR) → soft 検出器 (LLR) の変換。

LLR の符号規約は iq_model と同じ：L = log P(0)/P(1)、正なら 0 寄り。
検出器 = 2 測定の XOR なので、検出器の LLR は boxplus
    L1 ⊞ L2 = 2 artanh(tanh(L1/2) tanh(L2/2))
で与えられる。(6.0, 0.5) と (0.5, 6.0) が同じ値になる＝「どちらの測定が不確かだったか」を
検出器の値からは復元できない（plan01 §3.5、軸2の根拠）。

「古典的な検出器ベースの表現ではここまでしか情報を持てない」という上限を正しく体現するためのコード

"""
import numpy as np

from leakdec.sim.circuit_map import CircuitMap


def boxplus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """要素ごとの a ⊞ b。数値安定形（exp の引数は常に非正）。±inf も扱う。

    恒等式: a ⊞ b = sign(a)sign(b) min(|a|,|b|) + log1p(e^{-|a+b|}) - log1p(e^{-|a-b|})
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sgn = np.sign(a) * np.sign(b)
    core = sgn * np.minimum(np.abs(a), np.abs(b))
    with np.errstate(invalid="ignore"): # with内では、NaNが出ても警告を出さない。withを抜けたら警告がでるように戻す
        corr = np.log1p(np.exp(-np.abs(a + b))) - np.log1p(np.exp(-np.abs(a - b)))
    # 両方 ±inf のとき a±b が nan になる。極限は sign(a)sign(b)·inf
    both_inf = np.isinf(a) & np.isinf(b)
    limit = np.where(sgn > 0, np.inf, -np.inf) # sgn>0なら+inf、sgn<0なら-inf
    return np.where(both_inf, limit, core + np.where(np.isnan(corr), 0.0, corr))


# 補助量子ビットの生の測定LLRを検出器のLLRに変換すること。
def detector_llrs(aux_llr: np.ndarray, cmap: CircuitMap) -> np.ndarray:
    """補助測定 LLR (..., num_aux_measurements) → 検出器 LLR (..., num_detectors)。

    データ読み出しはどの検出器にも寄与しない（かつ論理観測量の材料＝ラベル）ので入力に含めない。
    """
    
    if aux_llr.shape[-1] != cmap.num_aux_measurements:
        raise ValueError(f"最後の軸は補助測定数 {cmap.num_aux_measurements} であること")
    # 「M2D(測定→検出器)対応表」から「各検出器がどの２つの補助測定のXORで作られるか」を取得する
    # 各検出器に対応する２つの補助測定の LLR を boxplus することで、検出器の LLR を計算する。
    pairs = cmap.detector_aux_measurements() # (num_detectors, 2) の配列。各検出器がどの補助測定のペアで構成されるかを示す。
    return boxplus(aux_llr[..., pairs[:, 0]], aux_llr[..., pairs[:, 1]])

"""
pairs = [0,4] → 測定0と測定4のXORで検出器0が作られる
"""

"""
測定が8箇所あるので、576個の補助測定。
それに対して、検出器は補助測定の隣り合う2つでXORすることで作られるので、
1補助量子ビットあたり検出器7個×72補助量子ビット = 504個の検出器が作られる。
 pairs = cmap.detector_aux_measurements() 
 ↑
 どの測定とどの測定が連続値ラウンドとしてペアになっているかのテーブル。
 同じ測定インデックスがpargsの中に2回出てくることがある。
"""