from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from house_prices.config import default_data_dir, default_output_path
from house_prices.ensemble import predict_sale_prices
from house_prices.preprocess import build_matrices

warnings.filterwarnings("ignore")


def run_training_pipeline(
    *,
    data_dir: Path | None = None,
    output_path: Path | None = None,
    run_cv: bool = True,
    verbose: bool = True,
) -> Path:
    """前処理から各モデル推論・ブレンド・提出 CSV 書き出しまで一気通貫。"""
    data_dir = data_dir or default_data_dir()
    output_path = Path(output_path or default_output_path())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"data_dir: {data_dir}")
        print("Building feature matrices...")
    X_train, X_test, y_train, test_id, skew_info = build_matrices(data_dir)
    if verbose:
        print(f"  skewed cols (box-cox): {len(skew_info)}")
        print(f"  X_train {X_train.shape}, X_test {X_test.shape}")

    y_pred, _ = predict_sale_prices(
        X_train, y_train, X_test, run_cv=run_cv, verbose=verbose
    )

    sub = pd.DataFrame({"Id": test_id, "SalePrice": y_pred})
    sub.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved: {output_path}")
        print(f"Mean SalePrice: ${y_pred.mean():,.0f}")
    return output_path
