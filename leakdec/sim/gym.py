# QEC環境　デコーダ研究のインタフェース、シンドロームを出す・予測を採点するロジックをまとめる
import sys  # 未使用だが将来の終了処理用に import
from typing import Tuple  # 戻り値の型ヒント
from random import randint  # Stim サンプラのシード生成

import numpy as np  # 行列演算・ブール配列
import stim  # 回路・DEM・サンプリング

from leakdec.sim.circuit_gen import get_72_12_6_code, get_144_12_12_code, GF2  # BB 符号生成と GF(2)


# TODO: See if its faster to do this with bool arrays with numba/cython
def f2_matmul(m1: np.ndarray, m2: np.ndarray):
    """
    GF(2) 上で行列積を計算する（加算は mod 2 = XOR）。
    """
    return ((m1.astype(np.uint8) @ m2.astype(np.uint8)) % 2).astype(np.bool_)


def parse_dem_to_decoding_matrices(
    dem: stim.DetectorErrorModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DEM をデコーディング用行列 H, L, P に変換する。

    NOTE: DEM は flatten 済み（REPEAT ブロックなし）であること。

    n = エラー機構の数, l = 論理オブザーバブル数, d = 検出器数
    - H: 時空パリティ行列 (d x n)。H(i,j)=1 ならエラー j が検出器 i を反転
    - L: 論理行列 (l x n)。L(i,j)=1 ならエラー j が論理オブザーバブル i を反転
    - P: 確率ベクトル (n,)。P[i] はエラー機構 i の発生確率
    """
    decoding_matrix = np.zeros((dem.num_detectors, dem.num_errors), dtype=np.bool_)
    logical_decoding_matrix = np.zeros(
        (dem.num_observables, dem.num_errors), dtype=np.bool_
    )
    probabilities = np.zeros((dem.num_errors), dtype=np.float32)

    error_index = 0  # 現在処理中のエラー機構の列インデックス
    for dem_instruction in dem:  # DEM の各行を走査
        type = dem_instruction.type  # 命令種別（error, detector_error 等）
        if type == "error":  # 独立エラー機構の行
            probabilities[error_index] = dem_instruction.args_copy()[0]  # 第1引数=確率
            for target in dem_instruction.targets_copy():  # フリップする検出器・論理オブザーバブル
                col = target.val  # ターゲット ID
                if target.is_relative_detector_id():  # 検出器 D#
                    decoding_matrix[col, error_index] = True
                elif target.is_logical_observable_id:  # 論理オブザーバブル L#
                    logical_decoding_matrix[col, error_index] = True
            error_index += 1  # 次のエラー機構へ

    return (decoding_matrix, logical_decoding_matrix, probabilities)


class QECGym:
    """
    強化学習の gym に倣った、ML 量子エラー訂正デコーダの学習・評価用環境。
    """

    def __init__(
        self,
        code_name: str,
        logical_operator: str,
        noise_type: str,
        physical_error_rate: float,
        num_rounds: int = -1,
        measure_both=False,
        load_saved_logical_ops=False,
    ) -> None:
        """
        QEC 環境を構築し、回路・DEM・デコーディング行列を初期化する。

        Args:
            code_name: 符号名 ('bbc-72-12-6' または 'bbc-144-12-12')
            logical_operator: 測定する論理演算子 ('X' または 'Z')
            noise_type: ノイズ種別 ('data' または 'circuit')
            physical_error_rate: 物理エラー率 p
            num_rounds: シンドロム測定ラウンド数（-1 なら距離と同じ）
            measure_both: True なら X/Z 両方の検出器を有効化
            load_saved_logical_ops: True なら保存済み論理 X 演算子を .npy から読み込む
        """
        self.code_name = code_name
        self.logical_operator = logical_operator
        self.noise_type = noise_type
        self.phyical_error_rate = physical_error_rate  #  typo: physical の綴りミス（既存コード踏襲）
        self.num_rounds = num_rounds
        if logical_operator == "X":  # 論理 X 実験 → X シンドロム検出器のみ
            x_detectors = True
            z_detectors = False
        elif logical_operator == "Z":  # 論理 Z 実験 → Z シンドロム検出器のみ
            z_detectors = True
            x_detectors = False
        if measure_both:  # 両方測る設定
            z_detectors = True
            x_detectors = True

        if "bbc" in code_name:  # 二変数バイシクル符号
            if code_name == "bbc-72-12-6":
                code = get_72_12_6_code()  # [[72,12,6]] 符号
                distance = 6
                if load_saved_logical_ops:
                    code.logical_X_ops = GF2(  # 事前計算した論理 X を上書き（高速化）
                        np.load(
                            "data/logical_ops/bbc-72-12-6_logical_X_ops.npy"
                        )
                    )
            elif code_name == "bbc-144-12-12":
                code = get_144_12_12_code()
                distance = 12
                if load_saved_logical_ops:
                    code.logical_X_ops = GF2(
                        np.load(
                            "data/logical_ops/bbc-144-12-12_logical_X_ops.npy"
                        )
                    )

            if noise_type == "circuit":  # 回路レベルノイズ
                if num_rounds < 0:
                    num_rounds = distance  # 既定は符号距離と同じラウンド数
                circuit = code.create_syndrome_measurement_circuit(
                    x_detectors,
                    z_detectors,
                    num_rounds,
                    logical_operator,
                    physical_error_rate,
                )
        else:
            raise ValueError("Invalid code name.")
        self._dem = circuit.detector_error_model()  # 検出器エラーモデル
        stim_rand_seed = randint(0, 2**64)  # サンプラ用乱数シード
        #print("Using random seed for stim:", stim_rand_seed)
        self._dem_sampler = self._dem.compile_sampler(seed=stim_rand_seed)  # 高速サンプラ
        (
            self._spacetime_H,  # 時空パリティチェック行列 H
            self._logical_decoding_matrix,  # 論理デコーディング行列 L
            self._channel_probabilities,  # 各エラー機構の確率 P
        ) = parse_dem_to_decoding_matrices(self._dem.flattened())  # REPEAT を展開した DEM から抽出

        self._det_data: np.ndarray | None = None  # 直近サンプルの検出器結果
        self._actual_errors: np.ndarray | None = None  # 直近サンプルの真のエラー
        self._measured_observables: np.ndarray | None = None  # 直近の論理測定結果
        self._circuit = circuit  # 生成した Stim 回路
        self.N_E = self._spacetime_H.shape[1]  # エラー機構の総数

    def get_detector_error_model(self) -> stim.DetectorErrorModel:
        """回路の検出器エラーモデル (DEM) を返す。"""
        return self._dem

    def get_spacetime_parity_check_matrix(self) -> np.ndarray:
        """時空パリティチェック行列 H を返す。"""
        return self._spacetime_H

    def get_channel_probabalities(self) -> np.ndarray:
        """各エラー機構の発生確率ベクトル P を返す（メソッド名の typo は既存踏襲）。"""
        return self._channel_probabilities

    def get_logical_decoding_matrix(self) -> np.ndarray:
        """論理デコーディング行列 L を返す。"""
        return self._logical_decoding_matrix

    def get_decoding_instances(
        self, shots: int, return_errors: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        DEM から shots 回分の検出器・論理測定・（任意で）真エラーをサンプルする。

        回路を毎回シミュレートせず DEM サンプラを使うため高速。

        Returns:
            det_data: (shots, num_detectors) 検出器の反転
            measured_observables: (shots, k) 論理測定が -1 か（True/False）
            actual_errors: return_errors=True のとき (shots, num_errors)
        """
        (
            self._det_data,
            self._measured_observables,
            self._actual_errors,
        ) = self._dem_sampler.sample(shots=shots, return_errors=return_errors)
        return self._det_data, self._measured_observables, self._actual_errors
    
    #predicted_errors @ L^T と measured_observables の XOR → 論理エラー率の元データ
    def evaluate_predictions(
        self,
        predicted_errors: np.ndarray,
        return_errors: bool = False,
        allow_inconsistent_predictions: bool = False,
        use_canonical_errors: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        予測エラーがシンドロム・論理測定と整合するか評価する。

        predicted_errors: (shots, num_errors) のブール配列
        """
        if use_canonical_errors:
            raise NotImplementedError
        predicted_syndromes = f2_matmul(predicted_errors, self._spacetime_H.T)  # H^T e
        differences = np.logical_xor(predicted_syndromes, self._det_data)  # シンドロム不一致
        inconsistencies = differences.sum(axis=1)  # ショットごとの不一致ビット数
        if not allow_inconsistent_predictions and inconsistencies.any():
            raise ValueError("Predicted errors are not consistent with the syndrome")
        inconsistencies = inconsistencies != np.zeros_like(inconsistencies)  # 0 以外なら True
        inconsistencies = np.expand_dims(inconsistencies, axis=1)  # (shots, 1) に拡張

        predicted_observables = f2_matmul(
            predicted_errors, self._logical_decoding_matrix.T
        )  # 予測論理オブザーバブル
        observable_difference = np.logical_xor(
            self._measured_observables, predicted_observables
        )  # 論理エラー（測定と予測の差）
        if return_errors:
            # 論理不一致 OR シンドローム不一致 → 誤りとカウント
            return observable_difference + inconsistencies, self._actual_errors
        else:
            return observable_difference, None

    def __repr__(self) -> str:
        # 対話環境で gym オブジェクトを表示するときの文字列
        return (
            f"QECGym(code_name={self.code_name},"
            f"logical_operator={self.logical_operator},"
            f"noise_type={self.noise_type},"
            f"physical_error_rate={self.phyical_error_rate})"
        )
