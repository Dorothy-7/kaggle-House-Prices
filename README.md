# Kaggle House Prices — Advanced Regression Techniques

## 目的（1 行）

[Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) において、ドメイン特徴量・歪度しきい値付き Box-Cox・**異系統モデルの役割分担つき**加重ブレンドで **Public Leaderboard RMSLE 0.12127** を達成した再現用リポジトリです。

### スコア（採用側が最初に見る想定で表に集約）

| 指標 | 値 | メモ |
|------|-----|------|
| **Public LB**（RMSLE） | **0.12127** | 提出ファイル [`outputs/submission_house_prices_best.csv`](outputs/submission_house_prices_best.csv) |
| **Private LB**（RMSLE） | （未記録） | Kaggle の Submissions タブの Private Score を転記してください（履歴が消えていなければ確認可能） |
| **CV RMSE**（`log1p(SalePrice)` 上） | **0.12491 ± 0.01102** | *LightGBM* 単体・5-fold・`shuffle=True`, `random_state=123`。`python run_house_prices.py` 実行時に `--no-cv` なしで標準出力にも出ます。LB の RMSLE と同じ log 差の定義です。 |

詳細な経緯・試行錯誤は [`docs/PORTFOLIO_house_prices.md`](docs/PORTFOLIO_house_prices.md) にあります。

---

## GitHub の「About」を埋める（一覧で信頼が上がる）

リポジトリの **Settings は触れない**ため、次を GitHub 上で手入力してください。

**Short description（例・コピペ可）**

```text
House Prices (Kaggle): feature engineering, Box-Cox on skewed numerics, weighted blend of GBDT + linear — Public LB RMSLE 0.12127.
```

**Topics（例）**

`python` `kaggle` `house-prices` `regression` `rmsle` `lightgbm` `xgboost` `catboost` `scikit-learn` `feature-engineering` `tabular-data`

---

## モデルの役割とブレンドの意図（「混ぜただけ」に見せない）

ファイル名を `submission_*_best.csv` に変更したのは、**スコア達成の説明責任をモデル設計側に寄せるため**です（旧名 `submission_boxcox_5models_ensemble.csv` は誤解を招きやすかった）。

| 系統 | 役割 | 本パイプラインでの立ち位置 |
|------|------|---------------------------|
| **LightGBM**（5 シード平均） | メインの非線形 GBDT。交互作用・しきい値を素直に捉える。 | 重み **0.35**。**CV の基準線**もここで計測。 |
| **XGBoost**（3 シード、浅めの木＋正則化） | LGB と別実装の GBDT。過学習しにくい設定で、誤差パターンが完全には重ならない。 | **0.22**。多様性確保。|
| **Ridge**（α=10,15,20 の平均） | L2 線形。スムーズな大域的傾向・スケーリング後の安定予測。 | **0.18**。|
| **Lasso**（小さめ α 3 本平均） | L1 線形。スパース化し「効いていない」方向の係数を絞る。**ブレンドでの補正項**に近い役割。 | **0.15**。|
| **CatBoost**（3 シード） | 別最適化のツリーブースティング。LGB/XGB とは勾配・分岐処理が異なり、再度 **多様性** を足す。 | **0.10**。|

**ブレンド係数（ログ予測の線形結合）**: `0.35·LGB + 0.22·XGB + 0.18·Ridge + 0.15·Lasso + 0.10·Cat`

**単体 Public LB を CSV で残していなかった**ため、リーダーボード上の「モデル別の厳密な一桁比較」は再掲できません。一方で、(1) CV では LGB が基準になり (2) アンサンブルは **相関の低い予測を足す**構成にしており、(3) **重みは複数 submission の比較**から決めた、という意味で「とりあえず混ぜた」ものではありません。過去実験ログの一例（**同一パイプラインではない**ので参考値）として、アーカイブ内ノートでは複数モデルを束ねたときに検証側スコアが単体ベストより有利になるケースを確認しています。

---

## 残差・誤差分析（面接で短く話せるネタ）

### 目的変数

- **`SalePrice`** は右裾が長い分布になりがち。学習・評価は **`y = log1p(SalePrice)`**（予測後 `expm1`）で、**RMSLE と整合**する。

### Box-Cox（`boxcox1p`, λ=0.15）

- 学習＋テストを結合した列ごとの **歪度の絶対値が 0.5 超**のみ変換。ベストノートの実行ログでは **39 列** が対象。
- 歪度が大きかった例（実行ログより、変換前の参考値）: **MiscVal ≈ 21.95**, **PoolArea ≈ 17.70**, **LotArea ≈ 13.12**, **LowQualFinSF ≈ 12.09** など。面積・金額寄りの長い裾を押さえ、**線形系と GBDT の両方が扱いやすいスケール**に寄せた。

### 有名外れ値の扱い

- 多くの解法で引用される例に沿い、**`GrLivArea > 4000` かつ `SalePrice < 300000`** の学習行を除外（面積は大きいが価格が著しく低い 2 点）。実装は [`src/house_prices/preprocess.py`](src/house_prices/preprocess.py) の `remove_grliv_outliers`。

### 評価関数（コード）

[`src/house_prices/metrics.py`](src/house_prices/metrics.py) に、log 空間 RMSE / 価格スケール RMSLE を集約。

---

## 実行

```bash
# リポジトリ直下で（venv を有効化したうえで）
python run_house_prices.py
# CV を省略して速くする場合
python run_house_prices.py --no-cv
```

データは `data/train.csv`・`data/test.csv` を [`data/`](data/) に置く必要があります。GitHub にはコンペデータを含めないため、[コンペの Data タブ](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)から取得してください。

### フォルダ

| 内容 | パス |
|------|------|
| 本番パッケージ（前処理・学習・ブレンド） | [`src/house_prices/`](src/house_prices/) |
| 学習実行のエントリ（薄い CLI） | [`run_house_prices.py`](run_house_prices.py) |
| ベスト構成の根拠ノート | [`notebooks/house_prices_best_public_0_12127.ipynb`](notebooks/house_prices_best_public_0_12127.ipynb) |
| パイプライン説明用ノート | [`notebooks/house_prices_pipeline.ipynb`](notebooks/house_prices_pipeline.ipynb) |
| 提出 CSV | [`outputs/submission_house_prices_best.csv`](outputs/submission_house_prices_best.csv) |
| 長文メモ | [`docs/PORTFOLIO_house_prices.md`](docs/PORTFOLIO_house_prices.md) |

### 開発（テスト・Lint）

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check src tests
pytest -q
```

CI: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)（Ruff + pytest。学習ライブラリは入れずユニットテストのみ）。

---

## 退避した実験

[`archive/README.txt`](archive/README.txt) を参照。旧ノートブック・試行提出 CSV は `archive/` 以下です。

---

## コミット方針（今後）

履歴はレビュー側の信頼にも効くため、**小さな論理単位でコミット**する方針です（ドキュメント／前処理／モデル／ノート／CI などを混ぜない）。
