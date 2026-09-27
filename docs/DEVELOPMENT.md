# 開発ガイド

このリポジトリで手を動かすときの手順書。**研究の内容と結果は [../README.md](../README.md) を見ること。**

| ドキュメント | 内容 |
|---|---|
| [../README.md](../README.md) | 研究の背景・成果・設計判断（読み物） |
| [progress_20260925.md](progress_20260925.md) | 全結果の数値、計画から修正した点、残タスク |
| [models_cheatsheet.md](models_cheatsheet.md) | B0〜B4 / ORC / ①〜④ / hybrid が何なのかの早見表 |
| [../RULES.md](../RULES.md) | 守るべき制約（過去に踏んだ地雷つき） |
| [../TODO.md](../TODO.md) | 実装 TODO と受け入れ基準、監査 16 項目 |
| [../CLAUDE.md](../CLAUDE.md) | Claude Code 用のプロジェクト文脈 |

---

## 実行のしかた

**必ず `paper2/` を作業ディレクトリにして実行する**（`data/`, `results/`, `runs/` を相対パスで読む）。

```bash
cd <このリポジトリのルート>
source ../.venv/bin/activate        # 仮想環境は親ディレクトリに置いている
```

初回のみ（`leakdec` を import 解決できるようにする）：

```bash
uv pip install --no-deps -e .
```

| やること | コマンド |
|---|---|
| テスト | `python -m pytest -q` |
| T1-6 関門 | `nohup python -u scripts/run_gate_t1_6.py > results/gate_t1_6.log 2>&1 &` |
| T2-1 一致検査 | `nohup python -u scripts/run_t2_1_check.py > results/t2_1_check.log 2>&1 &` |
| ベースライン B0〜B4/ORC | `nohup python -u scripts/run_t2_2.py [shots] [workers] [p_leak] [p_seep] > results/t2_2_<tag>.log 2>&1 &` |
| p 掃引 | `nohup python -u scripts/run_sweep_p.py --workers 12 > results/sweep_p/log 2>&1 &` |
| 学習 | `nohup python -u -m leakdec.learn.train --repr raw --out runs/r1_raw > runs/r1_raw.log 2>&1 &` |
| 評価（hybrid 含む） | `python -m leakdec.analysis.evaluate runs/r1_raw_seq --hybrid` |
| 図 | `python -m leakdec.analysis.plot_runs runs/r1_raw_seq runs/r2_hard_seq runs/r3_soft_seq runs/r4_raw_single` |
| 結果表 | `python -m leakdec.analysis.summarize_runs runs/r1_raw_seq runs/r2_hard_seq runs/r3_soft_seq runs/r4_raw_single` |

長時間ジョブは `nohup python -u`（`-u` 必須。`| tail` はバッファされて見えない）。

---

## ディレクトリ

```
paper2/
├── leakdec/                    ライブラリ本体
│   ├── sim/                    シミュレータ
│   │   ├── circuit_gen.py      BB 符号と Stim 回路の生成（myproject_copy から流用）
│   │   ├── gym.py              DEM → H, lx, priors, サンプリング（同上）
│   │   ├── shots.py            固定ショット集合（同上）
│   │   ├── circuit_map.py      測定 → 検出器の写像 M2D (504, 648) と測定の住所表  … T1-1
│   │   ├── iq_model.py         3 状態 IQ モデル f0/f1/f2（ガウス＋振幅減衰）      … T1-2
│   │   ├── leakage_model.py    リーク持続過程（2 状態マルコフ、真値 ℓ を出す）    … T1-3
│   │   ├── boxplus.py          検出器 LLR への合成 L1 ⊞ L2                       … T1-4
│   │   └── soft_syndrome.py    ショット生成。hard/soft/raw の 3 表現を同時に出す  … T1-5
│   ├── decode/                 古典デコーダと p_S の作り方（B0〜B4、ORC）
│   │   ├── bposd_baseline.py   BP-OSD ラッパ、式19 の事前確率更新、B0/B1        … T2-1
│   │   ├── leak_baselines.py   B2（Hanisch 円形境界）、B3（3 状態 MAP）、B3s     … T2-2
│   │   └── leak_hmm.py         B4（量子ビットごとの 2 状態 HMM、前向き後向き）
│   ├── learn/                  学習型デコーダ
│   │   ├── model.py            SyndromeTransformer（論理ヘッド＋リークヘッド）   … T3-2
│   │   ├── dataset.py          オンライン生成のデータローダ、3 表現の切り替え    … T3-1
│   │   └── train.py            学習ループ（CLI）                                … T3-3
│   └── analysis/               評価・作図
│       ├── evaluate.py         T2-2 と同一ショットでの評価、hybrid 復号（CLI）
│       ├── stats.py            Clopper-Pearson、McNemar、レイテンシの整形
│       ├── plot_runs.py        損失曲線・評価推移・P_L vs p・レイテンシ（CLI）
│       └── summarize_runs.py   ①〜④ を 1 表に（CLI）
├── scripts/                    使い捨ての実験ドライバ
│   ├── run_gate_t1_6.py        T1-6 関門（hard 対 soft の一致検査）
│   ├── run_t2_1_check.py       既存結果との一致検査
│   ├── run_t2_2.py             ベースライン B0〜B4/ORC の一括実行
│   └── run_sweep_p.py          物理エラー率 p の掃引
├── tests/                      pytest（約 80 件）
├── docs/                       progress_*.md, models_cheatsheet.md, わからないことメモ.txt
├── data/                       logical_ops/*.npy, circuit_map/*.npz（生成物）
├── results/                    ベースラインの結果、図、要約
├── runs/                       学習の run（config.json, best.pt, history.npz, eval_*.json）
├── saved_results/              固定ショット集合
├── CLAUDE.md RULES.md TODO.md README.md pyproject.toml
```

## 依存の向き

```
sim ──► decode ──► analysis
 └────► learn ─────┘
```

`sim` は他のどこにも依存しない。`decode` は `sim`（circuit_map, iq_model, leakage_model）だけを見る。
`learn` は `sim` を見る。`analysis` は全部を見る。**逆向きの import を作らないこと。**

## 環境

- 仮想環境：リポジトリの親ディレクトリの `.venv`（Python 3.10、torch 2.14.0+cu130）
- GPU：NVIDIA RTX 6000 Ada Generation
- 外部：`stim`, `galois`, `ldpc==2.4.1`, `numpy`, `scipy`, `soft_information_models`（editable）

---

## 整理の記録（2026-09-27）

フラット構成（`paper2/*.py` に 30 ファイル）からパッケージ構成に移した。**コードの中身は import 文と
docstring の実行例以外いっさい変えていない。** 検証：整理前に 86 件通過を記録し、整理後も同じ 86 件で確認した。

- `data/`, `results/`, `runs/`, `saved_results/` のパスは**すべて従来どおりプロジェクトルート相対**。
  過去の実行結果・チェックポイントはそのまま読める（`best.pt` は state_dict と args の素の dict なので
  モジュール位置の変更に影響されない）
- `results/t2_2.npz`（16 ショット・p_leak=0.02 の動作確認の残骸、監査項目 14）は
  `results/_smoketest_16shots_pl0.02.npz` に改名した。結果と紛らわしかったため

---

## Git 管理外のファイル

| ファイル | 理由 |
|---|---|
| `runs/**/*.pt` | 学習済み重み 15 MB × 8。再学習できる |
| `runs/**/eval_t22*.npz`, `t22_best.npz` | 評価の中間生成物。`evaluate` で再生成できる |
| `SCHEDULE.local.md` | 作業日程。個人の予定を含むため公開しない |
| `results/_smoketest_16shots_pl0.02.npz` | 16 ショットの動作確認の残骸 |

## 開発のときに踏んではいけない地雷（`RULES.md` の要点）

1. **BP-OSD は `schedule="serial"` 必須。** `parallel` は BP がほぼ収束せず LER が 15.6 倍悪化する
2. **`BpOsdDecoder` は 1 個で約 0.36 GB。** 複数同時保持すると OOM で落ちる。「パスごとにデコーダ 1 個」
3. **BB 符号では Λ（誤り抑制係数）を使わない。** 同じ多項式ファミリーでないため。$P_L$ を指標にする
4. **長時間ジョブは `nohup python -u`。** `-u` がないと出力がバッファされて進捗が見えない
5. **学習は停滞相（loss ≈ 0.686 が約 2 万 step）で見切らない。** 実測では約 1.5 万 step で抜けた
