"""Paper 2：BB 符号におけるリーク誤りの持続性を学習する soft デコーダ。

  leakdec.sim      … シミュレータ（回路生成・IQ モデル・リーク過程・ショット生成）
  leakdec.decode   … 古典デコーダとリーク前処理ベースライン（B0〜B4、ORC）
  leakdec.learn    … 学習型デコーダ（モデル・データローダ・学習ループ）
  leakdec.analysis … 評価・統計・作図

データと結果のパスはプロジェクトルート相対（data/, results/, runs/）なので、
スクリプトは必ず paper2/ を作業ディレクトリにして実行する。
"""
