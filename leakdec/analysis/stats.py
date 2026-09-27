"""評価の共通統計。0 誤りを LER=0 と書かない（Clopper-Pearson 95% 上側限界を併記、RULES R3-4）。"""
import numpy as np
from scipy.stats import beta as beta_dist


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    return 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))


def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    return 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1))


def ler_line(name: str, fails: int, n: int) -> str:
    return (f"{name}: {fails}/{n} = {fails / n:.4f}  "
            f"[CP95%: {cp_lower(fails, n):.4f}, {cp_upper(fails, n):.4f}]")


def latency_line(seconds: np.ndarray) -> str:
    return f"median {np.median(seconds):.4g} s, p99 {np.percentile(seconds, 99):.4g} s"


# AUDIT(2026-09-16) 【補強】不一致数だけ出して p 値を出していない。TODO では「3√(a+b)」の目安で有意と書いているので、
#   正確二項検定の p 値を併記できるようにする。
# FIX:
# from scipy.stats import binomtest
# def mcnemar_p(only_a: int, only_b: int) -> float:
#     """不一致 (only_a, only_b) に対する正確 McNemar（両側二項）p 値。"""
#     n = only_a + only_b
#     return 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
def mcnemar_counts(fail_a: np.ndarray, fail_b: np.ndarray) -> tuple[int, int]:
    """(A だけ失敗, B だけ失敗)。対応のある比較の要。"""
    return int((fail_a & ~fail_b).sum()), int((fail_b & ~fail_a).sum())
