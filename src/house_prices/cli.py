from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from house_prices.config import default_data_dir, default_output_path
from house_prices.ensemble import predict_sale_prices
from house_prices.preprocess import build_matrices

warnings.filterwarnings("ignore")


def run(
    *,
    data_dir: Path | None = None,
    output_path: Path | None = None,
    run_cv: bool = True,
    verbose: bool = True,
) -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="House Prices — Box-Cox + 5-model ensemble submission"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Kaggle data folder (train.csv / test.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip LightGBM cross-validation (faster)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less log output",
    )
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = default_output_path()

    run(
        data_dir=args.data_dir,
        output_path=out,
        run_cv=not args.no_cv,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
