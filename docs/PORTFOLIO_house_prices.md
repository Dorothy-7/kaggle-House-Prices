# Kaggle House Prices — ポートフォリオ用メモ

## 一言で

**住宅販売価格の回帰予測**（タブラーデータ）で、特徴量設計・歪度に基づく Box-Cox 変換・複数勾配ブースティング＋線形モデルの加重アンサンブルを組み合わせ、**Public Leaderboard で RMSLE 0.12127** を記録した取り組み。

---

## コンペ・データ

| 項目 | 内容 |
|------|------|
| 名称 | [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)（Kaggle） |
| 目的変数 | `SalePrice`（住宅販売価格） |
| 評価指標 | **RMSLE**（対数誤差の二乗平均の平方根。小さいほど良い） |
| 学習データ | 約 1,460 行 × 81 列（目的変数含む） |
| テストデータ | 約 1,459 行 |
| 提出ファイル | `outputs/submission_boxcox_5models_ensemble.csv` |
| **Public LB** | **0.12127** |

---

## 技術スタック

Python 3、pandas / NumPy、scikit-learn（`KFold`、`Ridge`、`Lasso`、`StandardScaler`）、LightGBM、XGBoost、CatBoost、SciPy（`skew`、`boxcox1p`）。

---

## パイプライン概要（ベスト提出）

実装の出典: Python パッケージ `src/house_prices/`（最高スコア時のノート: `notebooks/house_prices_best_public_0_12127.ipynb`）。

1. **目的変数**  
   `y = log1p(SalePrice)` で学習。予測後に `expm1` で元の価格スケールに戻す。

2. **外れ値の除去**  
   有名な例に沿い、`GrLivArea > 4000` かつ `SalePrice < 300000` の学習行を除外（面積は大きいのに価格が異常に低い点）。

3. **ドメイン特徴量**（例）  
   `QualGrLiv`、`TotalSF`、`QualTotalSF`、`Age`、`RemodAge`、`TotalBath`、`AreaPerRoom`、`GarageScore`、`Log_GrLivArea` / `Log_LotArea` / `Log_TotalSF`、`QualGrLivLog`、`TotalPorchSF`、`IsNew`、`IsRemodeled`、`HasBsmt`、`BsmtScore` など。

4. **欠損値**  
   数値は 0、カテゴリは `"None"` で埋める。

5. **Box-Cox（`boxcox1p`, λ=0.15）**  
   数値列のうち、学習＋テスト結合での**歪度の絶対値が 0.5 超**の列だけ変換（実行ログでは 39 列）。右裾が長い面積系などの分布を抑え、木モデル・線形モデルの両方に効きやすくする狙い。

6. **カテゴリ**  
   `pd.get_dummies(..., drop_first=True)` により One-Hot（実行時は特徴量 **283 次元**）。

7. **検証**  
   5-fold CV（`shuffle=True`, `random_state=123`）。ベストノートでは LightGBM 単体の fold RMSE 平均 **約 0.1249（±0.011）**（ログ空間の RMSE）。

8. **アンサンブル（5 系統の予測の加重平均 ※ログ空間）**  
   - **LightGBM**: シード 5 本（123, 456, 789, 42, 2024）の予測を平均  
   - **XGBoost**: シード 3 本の平均（`max_depth=3` など浅め＋正則化寄り）  
   - **Ridge**: α = 10, 15, 20 の 3 モデル平均（入力は `StandardScaler` 後）  
   - **Lasso**: α = 0.0005, 0.001, 0.0015 の 3 モデル平均  
   - **CatBoost**: シード 3 本、`depth=4`、`l2_leaf_reg=5` など  

   **最終ブレンド重み（ログ予測の線形結合）**  
   `0.35×LGB + 0.22×XGB + 0.18×Ridge + 0.15×Lasso + 0.10×CatBoost`

   LightGBM を軸に、勾配ブースティング同士の多様性（XGB / Cat）と線形（Ridge / Lasso）を足す構成。

---

## 試行錯誤の整理（考察）

同じ House Prices 向けに、リポジトリ内で段階的に試している。

- **単一モデル・単一シード**から、**LightGBM のみ複数シード平均**へ拡張（旧ノートは `archive/notebooks/root_experiments/` 以下）。分散低減として有効な定番手。
- **XGBoost との 2 モデル系**、**3〜4 モデル＋重み探索**（複数 `submission_w*.csv` は `archive/submissions/` に退避）など、**アンサンブルの寄与**を確認。
- **Box-Cox の有無**: 数値の歪度に応じた `boxcox1p` を導入。線形系とツリーの両方で扱いやすい特徴量にそろえる効果を期待し、**ベスト提出では Box-Cox ＋ 5 系統ブレンド**に収束。
- **ハイパーパラメータ**は、過学習抑制のため `num_leaves` や `learning_rate`、`subsample` / `colsample_bytree`、`reg_alpha` / `reg_lambda` を控えめに設定する方向でチューニング。

**学びの要約**

- タブラー住宅価格では、**ドメイン特徴量**と**目的変数の log 変換**が土台。
- **歪みの大きい数値列への Box-Cox** は、後段の線形モデルと相性が良く、**多様なモデルを混ぜるアンサンブル**と組み合わせやすい。
- **同系統モデルでもシード平均**、**異系統（GBDT / 線形 / CatBoost）の加重平均**で、Public LB の改善につながった。

---

## 再現時の参照

| 種別 | パス |
|------|------|
| 提出 CSV | `outputs/submission_boxcox_5models_ensemble.csv` |
| 本番コード | `src/house_prices/`（`ensemble.py` に重み・各モデル、`preprocess.py` に特徴量・Box-Cox） |
| 実行 | `python run_house_prices.py` または `PYTHONPATH=src python -m house_prices` |
| Jupyter | `notebooks/house_prices_pipeline.ipynb` |
| 旧ノート（参照用） | `archive/notebooks/root_experiments/最新12_archived.ipynb` |
| 生データ | `data/train.csv` 等（リポジトリ内 `data/`） |

※データは Kaggle から取得し、上記ディレクトリに配置する想定。

---

## 成果ハイライト

> Kaggle House Prices（回帰）に個人で参加し、特徴量エンジニアリング、歪度に基づく Box-Cox 変換、LightGBM・XGBoost・CatBoost・Ridge・Lasso の加重アンサンブルを実装。複数シード／複数ハイパーパラメータで予測を安定化し、Public Leaderboard（RMSLE）で **0.12127** を達成しました。

---

