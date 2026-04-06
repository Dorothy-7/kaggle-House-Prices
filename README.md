# Python3 — Kaggle House Prices ほか

## House Prices（本番パイプライン）

Kaggle [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 向けの再現コードです。

| 内容 | 場所 |
|------|------|
| パッケージ | `src/house_prices/` |
| 提出（ベスト） | `outputs/submission_boxcox_5models_ensemble.csv` |
| 説明・スコア | `docs/PORTFOLIO_house_prices.md` |

### 実行

```bash
# リポジトリ直下で（venv を有効化したうえで）
python run_house_prices.py
# CV を省略して速くする場合
python run_house_prices.py --no-cv
```

データは `data/train.csv`・`data/test.csv` が `data/` にある必要です。GitHub にはコンペデータを含めていないため、[コンペページ](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) から取得して `data/` に置いてください。

### フォルダ

- `docs/` … ポートフォリオ用メモ・列説明など（旧 `情報/`）
- `outputs/` … 提出 CSV（`run_house_prices.py` の既定出力先）

### 退避した実験ファイル

`archive/README.txt` を参照。旧ノートブック・試行提出 CSV は `archive/` 以下に移しています。
