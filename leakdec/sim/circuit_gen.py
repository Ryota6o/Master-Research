# 符号定義・Stim回路生成
"""
二変数バイシクル (bivariate bicycle, BB) 符号の Stim 回路を生成するモジュール。

多項式 x^i y^j を巡回シフト行列のテンソル積として表現し、
パリティチェック行列 Hx, Hz および論理演算子を構築する。
Stim 形式のシンドロム測定回路を出力する。
"""

import sys
from typing import List, Tuple, Literal, Sequence

import numpy as np

import stim
import galois
from galois import FieldArray

# GF(2) 上の演算（加算は XOR、乗算は AND の積の mod 2）
GF2 = galois.GF(2)


class Monomial:
    """
    トーラス上の単項式 x^{x_power} y^{y_power} を表すクラス。

    格子サイズは l × m。x は長さ l の巡回シフト、y は長さ m の巡回シフトに対応する。
    """

    def __init__(self, l, m, x_power, y_power) -> None:
        """
        Args:
            l: x 方向の巡回長（シフト行列のサイズ）
            m: y 方向の巡回長
            x_power: x の指数（0 以上 l 未満）
            y_power: y の指数（0 以上 m 未満）
        """
        self.l = l
        self.m = m
        self.x_power = x_power
        self.y_power = y_power

    def to_matrix(self) -> FieldArray:
        """単項式に対応する l*m × l*m の巡回行列（GF2）を返す。"""
        return GF2(
            np.kron(
                # x 方向に x_power 回シフトした行列(shift_matrixで作成した行列をx_power回掛け算する)
                np.linalg.matrix_power(shift_matrix(self.l), self.x_power),
                # y 方向に y_power 回シフトした行列
                np.linalg.matrix_power(shift_matrix(self.m), self.y_power),
            )
        )

    def to_vector(self) -> FieldArray:
        """単項式を l*m 次元のベクトル（Kronecker 積）として返す。"""
        first = np.zeros(self.l, dtype=np.uint8)
        second = np.zeros(self.m, dtype=np.uint8)
        first[self.x_power] = 1
        second[self.y_power] = 1
        return GF2(np.kron(first, second))

    def __str__(self) -> str:
        return f"x^{self.x_power} y^{self.y_power}"

    def __repr__(self) -> str:
        return f"Monomial({self.l}, {self.m}, {self.x_power}, {self.y_power})"

    def __mul__(self, other):
        """単項式の積: 指数はそれぞれ mod l, mod m で加算。"""
        assert self.l == other.l
        assert self.m == other.m
        return Monomial(
            self.l,
            self.m,
            (self.x_power + other.x_power) % self.l,
            (self.y_power + other.y_power) % self.m,
        )

    def __eq__(self, other):
        if (
            self.l != other.l
            or self.m != other.m
            or self.x_power != other.x_power
            or self.y_power != other.y_power
        ):
            return False
        return True

    def inverse(self):
        """逆単項式 x^{-x_power} y^{-y_power}（巡回群上での逆元）。"""
        return Monomial(
            self.l,
            self.m,
            (self.l - self.x_power) % self.l,
            (self.m - self.y_power) % self.m,
        )


class Polynomial:
    """
    GF(2) 上の多項式（複数単項式の和）を l*m 次元ベクトルで表現するクラス。

    インデックス i = x_power * m + y_power で係数 1 の項を格納する。
    """

    def __init__(
        self,
        monomials: Sequence[Monomial],
        l: int = 0,
        m: int = 0,
        vec: FieldArray | None = None,
    ) -> None:
        """
        単項式リストから構築するか、既存の係数ベクトル vec から復元する。

        Args:
            monomials: 構成する単項式のリスト（空のときは l, m 必須）
            l, m: 格子サイズ（monomials が空のときのみ使用）
            vec: 既知の係数ベクトル（指定時は monomials から vec を再構築しない）
        """
        if len(monomials) == 0:
            assert l != 0
            assert m != 0
            self.l = l
            self.m = m
        else:
            self.l = monomials[0].l
            self.m = monomials[0].m
        if vec is None:
            # 単項式リストから係数ベクトルを構築
            self.vec = GF2(np.zeros(self.l * self.m, dtype=np.uint8))
            self.monomials: Sequence[Monomial] = monomials
            for i in monomials:
                self.vec[i.x_power * self.m + i.y_power] = 1
        else:
            # ベクトルから非ゼロ項を単項式に復元
            self.vec = vec
            if len(monomials) == 0:
                self.monomials = []
                nonzero_vals = vec.nonzero()[0]
                for i in nonzero_vals:
                    x_power = i // m
                    y_power = i - m * x_power
                    self.monomials.append(Monomial(l, m, x_power, y_power))

    def __str__(self) -> str:
        monomial_strings = [str(m) for m in self.monomials]
        return "Polynomial: " + " + ".join(monomial_strings)

    def __mul__(self, other):
        """多項式積（GF2）: 各単項式ペアの積の和。"""
        vec = GF2(np.zeros(self.l * self.m, dtype=np.uint8))
        for m in self.monomials:
            for o in other.monomials:
                x_power = (m.x_power + o.x_power) % self.l
                y_power = (m.y_power + o.y_power) % self.m
                vec[x_power * self.m + y_power] += GF2(1)
        return Polynomial([], self.l, self.m, vec)

    def __add__(self, other):
        """多項式和（GF2）: ベクトルの成分ごと XOR。"""
        return Polynomial([], self.l, self.m, self.vec + other.vec)

    def __eq__(self, other):
        if (
            self.l != other.l
            or self.m != other.m
            or not np.array_equal(self.vec, other.vec)
            or self.monomials != other.monomials
        ):
            return False

        return True

    def T(self):
        """転置に相当する操作: 各単項式を逆元に置き換える。"""
        if len(self.monomials) == 0:
            return self
        return Polynomial([m.inverse() for m in self.monomials])


class Code:
    """
    二変数バイシクル符号の定義と Stim 回路生成を担うクラス。

    多項式 A, B から X/Z スタビライザのパリティチェック行列を構築し、
    論理演算子を求めたうえでシンドロム測定回路を生成する。
    """

    def __init__(
        self,
        A_polynomial: Polynomial,
        B_polynomial: Polynomial,
        f: Polynomial | None = None,
    ) -> None:
        """
        Args:
            A_polynomial: 左ブロックに現れる多項式 A
            B_polynomial: 右ブロックに現れる多項式 B
            f: 論理演算子探索用の候補多項式 f（None なら自動探索）
        """
        self.l = A_polynomial.monomials[0].l
        self.m = A_polynomial.monomials[0].m
        self.A_polynomial = A_polynomial
        self.B_polynomial = B_polynomial
        # Hx: X 型スタビライザのチェック行列 [A | B] の各項を行列和で構築
        self.Hx = GF2(np.zeros((self.l * self.m, 2 * self.l * self.m), dtype=np.uint8))
        # Hz: Z 型スタビライザ（Hx の転置構造に対応）
        self.Hz = GF2(np.zeros((self.l * self.m, 2 * self.l * self.m), dtype=np.uint8))
        for i in A_polynomial.monomials:
            self.Hx[:, : self.l * self.m] += i.to_matrix()
            self.Hz[:, self.l * self.m :] += i.to_matrix().T
        for i in B_polynomial.monomials:
            self.Hx[:, self.l * self.m :] += i.to_matrix()
            self.Hz[:, : self.l * self.m] += i.to_matrix().T

        # CSS 条件: Hx と Hz の核の次元が一致することを確認
        assert self.Hx.shape[0] - np.linalg.matrix_rank(self.Hx) == self.Hz.shape[
            0
        ] - np.linalg.matrix_rank(self.Hz)
        # 論理量子ビット数 k = 2 * (データ量子ビット数 - ランク)
        self.k = 2 * (self.Hx.shape[0] - np.linalg.matrix_rank(self.Hx))
        # 全 k 個の論理 X/Z 演算子（2klm ビットのサポートベクトル）を返す。
        self.logical_X_ops, self.logical_Z_ops = self.get_logical_operators(f)

    def check_X_op_commutes(self, poly_1: Polynomial, poly_2: Polynomial) -> bool:
        """
        X 型演算子 (poly_1 左, poly_2 右) が Z スタビライザと可換か判定する。
        """
        vec1 = poly_1.vec
        vec2 = poly_2.vec
        for i in range(self.l * self.m):
            left_parity = int(np.dot(self.Hz[i, : self.l * self.m], vec1))
            right_parity = int(np.dot(self.Hz[i, self.l * self.m :], vec2))
            if (left_parity + right_parity) % 2 != 0:
                return False
        return True

    def check_Z_op_commutes(self, poly_1: Polynomial, poly_2: Polynomial) -> bool:
        """
        Z 型演算子 (poly_1 左, poly_2 右) が X スタビライザと可換か判定する。
        """
        vec1 = poly_1.vec
        vec2 = poly_2.vec
        for i in range(self.l * self.m):
            left_parity = int(np.dot(self.Hx[i, : self.l * self.m], vec1))
            right_parity = int(np.dot(self.Hx[i, self.l * self.m :], vec2))
            if (left_parity + right_parity) % 2 != 0:
                return False
        return True

    def _find_f_spanning_sets(
        self, f_cands: FieldArray, f_cand: Polynomial | None
    ) -> Tuple[list[FieldArray], list[FieldArray]]:
        """
        論理演算子の形 {X(αf, 0) | α ∈ M} と {Z(0, αf^T) | α ∈ M} について、
        それぞれ k/2 論理量子ビット分を張る演算子集合を探す。

        Returns:
            (X1_ops, Z2_ops): 各 l*m 個の候補行ベクトル（big 行列の拡張行）
        """
        Hx_rank = np.linalg.matrix_rank(self.Hx)
        Hz_rank = np.linalg.matrix_rank(self.Hz)
        for i in range(f_cands.shape[0]):
            if f_cand is None:
                f_cand = Polynomial([], self.l, self.m, GF2(f_cands[i, :]))
            # f は B の零空間に属する: f*B = 0
            assert f_cand * self.B_polynomial == Polynomial([], self.l, self.m)
            assert self.check_X_op_commutes(f_cand, Polynomial([], self.l, self.m))
            assert self.check_Z_op_commutes(Polynomial([], self.l, self.m), f_cand.T())
            # すべての単項式 α に対する αf, αf^T の係数ベクトル
            all_f_ops = GF2(
                np.zeros((self.l * self.m, f_cands.shape[1]), dtype=np.uint8)
            )
            all_f_T_ops = GF2(np.zeros_like(all_f_ops))
            for ii, alpha in enumerate(
                [
                    Monomial(self.l, self.m, x_pow, y_pow)
                    for x_pow in range(self.l)
                    for y_pow in range(self.m)
                ]
            ):
                all_f_ops[ii, :] = (Polynomial([alpha]) * f_cand).vec
                all_f_T_ops[ii, :] = (Polynomial([alpha]) * f_cand.T()).vec
            # Hx に論理 X 候補を積み増しした拡大行列
            big_X_matrix = GF2(
                np.vstack((self.Hx, np.hstack((all_f_ops, np.zeros_like(all_f_ops)))))
            )
            big_Z_matrix = GF2(
                np.vstack(
                    (self.Hz, np.hstack((np.zeros_like(all_f_T_ops), all_f_T_ops)))
                )
            )
            # スタビライザを除いた独立な論理演算子がちょうど k/2 個あるか
            if (
                np.linalg.matrix_rank(big_X_matrix) - Hx_rank == self.k // 2
                and np.linalg.matrix_rank(big_Z_matrix) - Hz_rank == self.k // 2
            ):
                X1_ops = [
                    GF2(big_X_matrix[i + self.l * self.m, :])
                    for i in range(self.l * self.m)
                ]
                Z2_ops = [
                    GF2(big_Z_matrix[i + self.l * self.m, :])
                    for i in range(self.l * self.m)
                ]
                return X1_ops, Z2_ops
        sys.exit("Wasn't able to find f candidate")

    def _find_gh_spanning_sets(self, gh_cands):
        """
        論理演算子の形 {X(αg, αh) | α ∈ M} と {Z(αh^T, αg^T) | α ∈ M} について、
        それぞれ k/2 論理量子ビット分を張る演算子集合を探す。

        Returns:
            (X2_ops, Z1_ops): 拡大行列の末尾行から取り出した候補
        """
        Hx_rank = np.linalg.matrix_rank(self.Hx)
        Hz_rank = np.linalg.matrix_rank(self.Hz)
        for j in range(gh_cands.shape[0]):
            g_cand = Polynomial([], self.l, self.m, gh_cands[j, : self.l * self.m])
            h_cand = Polynomial([], self.l, self.m, gh_cands[j, self.l * self.m :])
            assert self.check_X_op_commutes(g_cand, h_cand)
            # g*B + h*A = 0（結合零空間の条件）
            assert (
                g_cand * self.B_polynomial + h_cand * self.A_polynomial
                == Polynomial([], self.l, self.m)
            )
            assert self.check_Z_op_commutes(h_cand.T(), g_cand.T())
            all_gh_ops = GF2(
                np.zeros((self.l * self.m, gh_cands.shape[1]), dtype=np.uint8)
            )
            all_hg_T_ops = GF2(np.zeros_like(all_gh_ops))
            for jj, alpha in enumerate(
                [
                    Monomial(self.l, self.m, x_pow, y_pow)
                    for x_pow in range(self.l)
                    for y_pow in range(self.m)
                ]
            ):
                all_gh_ops[jj, :] = GF2(
                    np.hstack(
                        (
                            (Polynomial([alpha]) * g_cand).vec,
                            (Polynomial([alpha]) * h_cand).vec,
                        )
                    )
                )
                all_hg_T_ops[jj, :] = GF2(
                    np.hstack(
                        (
                            (Polynomial([alpha]) * h_cand.T()).vec,
                            (Polynomial([alpha]) * g_cand.T()).vec,
                        )
                    )
                )
            big_X_matrix = GF2(np.vstack((self.Hx, all_gh_ops)))
            big_Z_matrix = GF2(np.vstack((self.Hz, all_hg_T_ops)))
            if (
                np.linalg.matrix_rank(big_X_matrix) - Hx_rank == self.k // 2
                and np.linalg.matrix_rank(big_Z_matrix) - Hz_rank == self.k // 2
            ):
                X2_ops = [GF2(big_X_matrix[-i - 1, :]) for i in range(self.l * self.m)]
                Z1_ops = [GF2(big_Z_matrix[-i - 1, :]) for i in range(self.l * self.m)]
                return X2_ops, Z1_ops
        sys.exit(1)

    def find_logical_qubits(
        self,
        X_ops: List[FieldArray],
        Z_ops: List[FieldArray],
        num_qubits: int,
        logical_ops: List[Tuple[FieldArray, FieldArray]],
    ) -> List[Tuple[FieldArray, FieldArray]]:
        """
        k/2 論理量子ビット分を張る X/Z 演算子リストから、
        対易関係を満たすペア (X_i, Z_i) を再帰的に選ぶ（シンプレクティック基底化）。

        Args:
            X_ops, Z_ops: 候補演算子（破壊的に pop される）
            num_qubits: 残り選ぶ論理量子ビット数
            logical_ops: 既に確定したペアのリスト

        Returns:
            長さ num_qubits の (X, Z) ペアのリスト
        """
        if num_qubits == 0:
            return logical_ops
        X1 = X_ops.pop(0)
        Z1_index = -1
        for i, Z_op in enumerate(Z_ops):
            if np.dot(X1, Z_op) == GF2(1):
                if Z1_index == -1:
                    Z1_index = i
                else:
                    # 既に見つかった Z1 と反可換な Z は Z1 に寄せて消す
                    Z_ops[i] = Z_op + Z_ops[Z1_index]
        if Z1_index == -1:
            # X1 はスタビライザに含まれる → 捨てて次の候補へ
            aug_mat = np.vstack((self.Hx, X1))
            assert np.linalg.matrix_rank(aug_mat) == np.linalg.matrix_rank(self.Hx)
            return self.find_logical_qubits(X_ops, Z_ops, num_qubits, logical_ops)
        Z1 = Z_ops.pop(Z1_index)
        for i, X_op in enumerate(X_ops):
            if np.dot(X_op, Z1) == GF2(1):
                X_ops[i] = X_op + X1

        logical_ops.append((X1, Z1))
        return self.find_logical_qubits(X_ops, Z_ops, num_qubits - 1, logical_ops)

    def get_logical_operators(
        self,
        f_cand: Polynomial | None = None,
    ) -> Tuple[FieldArray, FieldArray]:
        """
        全 k 個の論理 X/Z 演算子（2k*l*m ビットのサポートベクトル）を返す。

        unprimed: f 系の X1 と gh 系の Z1、primed: gh 系の X2 と f 系の Z2。
        """
        f_cands, gh_cands = self._get_logical_op_polynomial_cands()
        X1_ops, Z2_ops = self._find_f_spanning_sets(f_cands, f_cand)
        X2_ops, Z1_ops = self._find_gh_spanning_sets(gh_cands)
        unprimed_ops = self.find_logical_qubits(X1_ops, Z1_ops, self.k // 2, [])
        primed_ops = self.find_logical_qubits(X2_ops, Z2_ops, self.k // 2, [])
        assert len(unprimed_ops) == self.k // 2 and len(primed_ops) == self.k // 2
        logical_X_ops = np.vstack(
            [ops[0] for ops in unprimed_ops] + [ops[0] for ops in primed_ops]
        )
        logical_Z_ops = np.vstack(
            [ops[1] for ops in unprimed_ops] + [ops[1] for ops in primed_ops]
        )
        return GF2(logical_X_ops), GF2(logical_Z_ops)

    def _get_logical_op_polynomial_cands(self) -> Tuple[FieldArray, FieldArray]:
        """
        論理演算子多項式の候補空間を零空間から取得する。

        f_cands: B^T の零空間（f*B=0 を満たす f）
        gh_cands: [B^T | A^T] の零空間（g*B + h*A = 0）
        """
        A = self.Hx[:, : self.l * self.m]
        B = GF2(self.Hx[:, self.l * self.m :])
        f_cands = B.T.null_space()
        tmp = GF2(np.hstack((B.T, A.T)))
        gh_cands = tmp.null_space()
        return f_cands, gh_cands

    def create_syndrome_measurement_circuit(
        self,
        x_detectors: bool,
        z_detectors: bool,
        num_rounds: int,
        logical_operator: Literal["X", "Z"],
        error_rate: float,
    ) -> stim.Circuit:
        """
        シンドロム測定 + データ測定 + 論理オブザーバブルを含む Stim 回路を生成する。

        Args:
            x_detectors: X シンドロ-ム用 DETECTOR を付けるか
            z_detectors: Z シンドロ-ム用 DETECTOR を付けるか
            num_rounds: ノイズ付きシンドロ-ム測定ラウンドの繰り返し回数
            logical_operator: 最終測定する論理演算子 ("X" または "Z")
            error_rate: ゲート・測定・アイドル時のデポーラ化確率

        Returns:
            完成した stim.Circuit
        """
        # A, B は各 3 項の多項式と仮定（BB 符号の標準形）
        A1 = self.A_polynomial.monomials[0].to_matrix()
        A2 = self.A_polynomial.monomials[1].to_matrix()
        A3 = self.A_polynomial.monomials[2].to_matrix()
        B1 = self.B_polynomial.monomials[0].to_matrix()
        B2 = self.B_polynomial.monomials[1].to_matrix()
        B3 = self.B_polynomial.monomials[2].to_matrix()
        circuit = stim.Circuit()
        qubits_per_block = self.l * self.m # [[72,12,6]]の場合、l: 6, m: 6 なので、qubits_per_block: 36
        # 4 ブロック: 左データ L, 右データ R, X 補助, Z 補助
        L_block = [i for i in range(qubits_per_block)]
        R_block = [i + qubits_per_block for i in range(qubits_per_block)]
        X_block = [i + 2 * qubits_per_block for i in range(qubits_per_block)]
        Z_block = [i + 3 * qubits_per_block for i in range(qubits_per_block)]

        # 初期状態: 論理 Z 測定なら |0⟩, 論理 X 測定なら |+⟩
        if logical_operator == "Z":
            circuit.append("RZ", L_block + R_block)
        elif logical_operator == "X":
            circuit.append("RX", L_block + R_block)

        # 第 1 ラウンド（検出器なし・エラーなし）
        circuit += self._syndrome_measurement_rounds(
            False,
            False,
            1,
            A1, A2, A3, B1, B2, B3,
            R_block,
            L_block,
            X_block,
            Z_block,
            0,
        )
        if num_rounds != 0:
            # メインの繰り返しラウンド（シンドロム検出器 + ノイズ）
            circuit += self._syndrome_measurement_rounds(
                x_detectors,
                z_detectors,
                num_rounds,
                A1, A2, A3, B1, B2, B3,
                R_block,
                L_block,
                X_block,
                Z_block,
                error_rate,
            )
        else:
            # ラウンド 0 のときはデータ量子ビットのみデポーラ化
            circuit.append("DEPOLARIZE1", L_block + R_block, error_rate)
        # 最終ラウンド（境界条件用、検出器は前ラウンドとの比較用）
        circuit += self._syndrome_measurement_rounds(
            x_detectors,
            z_detectors,
            1,
            A1, A2, A3, B1, B2, B3,
            R_block,
            L_block,
            X_block,
            Z_block,
            0,
        )
        # データ量子ビットを測定し、論理オブザーバブルを rec ターゲットで登録
        if logical_operator == "X":
            circuit.append("MX", L_block + R_block)
            ops = self.logical_X_ops
        else:
            circuit.append("MZ", L_block + R_block)
            ops = self.logical_Z_ops # logical_Z_ops は論理演算子を表す行列(行：論理ビット　列：物理ビット)
        for i in range(self.k):
            support = ops[i, :].nonzero()[0] # i番目の行で0ではない要素のインデックスを取得 [0,2]みたいな
            # 今回MX or MZで測定した量子ビット数は2×L×M個([[144,12,12]]の場合は2×12×6=144個) よって、 2lm - s というインデックスに配置される。
            targets = [2 * self.l * self.m - s for s in support]
            
            # stim.target_rec(-t):末尾からt番目の測定結果を参照　OBSERVABLE_INCLUDE: それらの測定結果のパリティをi番目の論理観測量として登録
            # 末尾から何番目といいう相対指定に変換する必要がある。物理ビットi番目の測定結果を格納する。
            circuit.append(
                "OBSERVABLE_INCLUDE", [stim.target_rec(-t) for t in targets], i
            )

        return circuit

    def _add_gates_round1(
        self,
        circuit: stim.Circuit,
        A1: np.ndarray,
        R_block: List[int],
        L_block: List[int],
        Z_block: List[int],
        error_rate: float = 0.0,
    ) -> stim.Circuit:
        """
        シンドロム測定の第 1 段: A1^T に従い R → Z ブロックへ CNOT。
        """
        cnot_qubits = []
        cnot_control_indices = A1.T.nonzero()[1]
        for i in range(self.l * self.m):
            cnot_qubits.append(R_block[cnot_control_indices[i]])
            cnot_qubits.append(Z_block[i])
        circuit.append("CNOT", cnot_qubits)
        if error_rate != 0:
            circuit.append("DEPOLARIZE2", cnot_qubits, error_rate)
            circuit.append("DEPOLARIZE1", L_block, error_rate)
        circuit.append("TICK")
        return circuit

    def _add_cnot_gates(
        self,
        circuit: stim.Circuit,
        X_block_target_indices: np.ndarray,
        Z_block_control_indices: np.ndarray,
        X_block_targets: list[int],
        Z_block_controls: list[int],
        X_block: list[int],
        Z_block: list[int],
        error_rate: float = 0.0,
    ) -> stim.Circuit:
        """
        1 タイムステップ分の CNOT 列を追加する。

        各データ位置 i について:
          X_block[i] → X_block_targets[...]（X 補助からデータへ）
          Z_block_controls[...] → Z_block[i]（データから Z 補助へ）
        """
        nonzero_X_block_target_indices = X_block_target_indices.nonzero()[1]
        nonzero_Z_block_control_indices = Z_block_control_indices.nonzero()[1]
        cnot_qubits = []
        for i in range(self.l * self.m):
            cnot_qubits.append(X_block[i])
            cnot_qubits.append(X_block_targets[nonzero_X_block_target_indices[i]])
            cnot_qubits.append(Z_block_controls[nonzero_Z_block_control_indices[i]])
            cnot_qubits.append(Z_block[i])
        circuit.append("CNOT", cnot_qubits)
        if error_rate != 0:
            circuit.append("DEPOLARIZE2", cnot_qubits, error_rate)
        circuit.append("TICK")
        return circuit

    def _add_gates_round7(
        self,
        circuit: stim.Circuit,
        A3: np.ndarray,
        R_block: list[int],
        L_block: list[int],
        X_block: list[int],
        error_rate: float = 0.0,
    ) -> stim.Circuit:
        """
        シンドロム測定の第 7 段: A3 に従い X 補助 → L データへ CNOT。
        """
        nonzero_X_block_target_indices = A3.nonzero()[1]
        cnot_qubits = []
        for i in range(self.l * self.m):
            cnot_qubits.append(X_block[i])
            cnot_qubits.append(L_block[nonzero_X_block_target_indices[i]])
        circuit.append("CNOT", cnot_qubits)
        if error_rate != 0:
            circuit.append("DEPOLARIZE2", cnot_qubits, error_rate)
            circuit.append("DEPOLARIZE1", R_block, error_rate)
        return circuit

    def _syndrome_measurement_rounds(
        self,
        x_detectors: bool,
        z_detectors: bool,
        rounds: int,
        A1: list[int],
        A2: list[int],
        A3: list[int],
        B1: list[int],
        B2: list[int],
        B3: list[int],
        R_block: list[int],
        L_block: list[int],
        X_block: list[int],
        Z_block: list[int],
        error_rate: float,
    ) -> stim.CircuitRepeatBlock:
        """
        1 ラウンド分のシンドロム測定サブ回路を構築し、rounds 回繰り返す。

        7 段の CNOT 列（A1, A2/A3, B2/B1, B1/B2, B3, A1/A2, A3）のあと
        Z/X 補助を測定。オプションで DETECTOR を前ラウンドとの差分に設定。
        """
        X_block_measured = X_block.copy()
        Z_block_measured = Z_block.copy()

        circuit_block = stim.Circuit()
        circuit_block.append("RZ", Z_block_measured)
        if error_rate > 0:
            circuit_block.append("X_ERROR", Z_block_measured, error_rate)
        circuit_block.append("TICK")
        circuit_block.append("RX", X_block_measured)
        if error_rate > 0:
            circuit_block.append("Z_ERROR", X_block_measured, error_rate)

        circuit_block = self._add_gates_round1(
            circuit_block, A1, R_block, L_block, Z_block, error_rate
        )
        circuit_block = self._add_cnot_gates(
            circuit_block,
            A2,
            A3.T,
            L_block,
            R_block,
            X_block,
            Z_block,
            error_rate,
        )
        circuit_block = self._add_cnot_gates(
            circuit_block,
            B2,
            B1.T,
            R_block,
            L_block,
            X_block,
            Z_block,
            error_rate,
        )
        circuit_block = self._add_cnot_gates(
            circuit_block,
            B1,
            B2.T,
            R_block,
            L_block,
            X_block,
            Z_block,
            error_rate,
        )
        circuit_block = self._add_cnot_gates(
            circuit_block,
            B3,
            B3.T,
            R_block,
            L_block,
            X_block,
            Z_block,
            error_rate,
        )
        circuit_block = self._add_cnot_gates(
            circuit_block,
            A1,
            A2.T,
            L_block,
            R_block,
            X_block,
            Z_block,
            error_rate,
        )
        circuit_block = self._add_gates_round7(
            circuit_block, A3, R_block, L_block, X_block, error_rate
        )
        if error_rate > 0:
            circuit_block.append("MZ", Z_block_measured, error_rate)
        else:
            circuit_block.append("MZ", Z_block_measured)
        circuit_block.append("TICK")
        if error_rate > 0:
            circuit_block.append("MX", X_block_measured, error_rate)
            circuit_block.append("DEPOLARIZE1", R_block + L_block, error_rate)
        else:
            circuit_block.append("MX", X_block_measured)

        if x_detectors or z_detectors:
            # 検出器座標の第 2 成分をラウンド番号にずらす
            circuit_block.append_from_stim_program_text("SHIFT_COORDS(0, 1)")
            offset = len(Z_block_measured) + len(X_block_measured)
            if z_detectors:
                for i in range(len(Z_block_measured)):
                    circuit_block.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-offset + i),
                            stim.target_rec(-2 * offset + i),
                        ],
                        (i, 0),
                    )
            if x_detectors:
                coord_offset = 0
                if z_detectors:
                    coord_offset = len(Z_block_measured)
                for i in range(len(X_block_measured)):
                    circuit_block.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-len(X_block_measured) + i),
                            stim.target_rec(-len(X_block_measured) - offset + i),
                        ],
                        (i + coord_offset, 0),
                    )

        return circuit_block * rounds

"""特定のクラスやオブジェクトに属さず、単体で呼び出し可能な関数"""
########################
# スタンドアロン関数   #
########################


def shift_matrix(l: int) -> FieldArray:
    """
    サイズ l×l の巡回シフト行列を GF(2) 上で生成する。

    例 (l=3):
        [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]]
    """
    s = np.diag(np.ones(l - 1, dtype=np.uint8), 1)
    s[l - 1, 0] = 1
    return GF2(s)

# ここのコード生成については合っている
def get_72_12_6_code(use_paper_f=False):
    """
    [[72, 12, 6]] BB 符号（l=m=6）の Code インスタンスを返す。

    Args:
        use_paper_f: True なら論文で与えられた具体的な f 多項式を使用

    Returns:
        Code オブジェクト（k=12, n=72 に相当するパラメータ）
    """
    a1 = Monomial(6, 6, 3, 0)
    a2 = Monomial(6, 6, 0, 1)
    a3 = Monomial(6, 6, 0, 2)
    A_poly = Polynomial([a1, a2, a3])

    b1 = Monomial(6, 6, 0, 3)
    b2 = Monomial(6, 6, 1, 0)
    b3 = Monomial(6, 6, 2, 0)
    B_poly = Polynomial([b1, b2, b3])

    if use_paper_f:
        f1 = Monomial(6, 6, 0, 3)
        f2 = Monomial(6, 6, 2, 0)
        f3 = Monomial(6, 6, 3, 3)
        f4 = Monomial(6, 6, 4, 0)
        f5 = Monomial(6, 6, 4, 3)
        f6 = Monomial(6, 6, 5, 3)
        f = Polynomial([f1, f2, f3, f4, f5, f6])
    else:
        f = None
    
    return Code(A_poly, B_poly, f)


def get_144_12_12_code():
    """
    [[144, 12, 12]] BB 符号（l=12, m=6）の Code インスタンスを返す。

    A, B の多項式は 72 符号と同型の指数配置をより大きな格子に拡張したもの。
    """
    l = 12
    m = 6

    a1 = Monomial(l, m, 3, 0)
    a2 = Monomial(l, m, 0, 1)
    a3 = Monomial(l, m, 0, 2)
    A_poly = Polynomial([a1, a2, a3])

    b1 = Monomial(l, m, 0, 3)
    b2 = Monomial(l, m, 1, 0)
    b3 = Monomial(l, m, 2, 0)
    B_poly = Polynomial([b1, b2, b3])
    return Code(A_poly, B_poly)
