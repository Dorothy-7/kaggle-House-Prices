from __future__ import annotations

import argparse
from pathlib import Path

from house_prices.config import default_data_dir, default_output_path
from house_prices.pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="House Prices — Box-Cox 前処理＋複数モデル加重ブレンド提出"
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

    out = args.output if args.output is not None else default_output_path()

    run_training_pipeline(
        data_dir=args.data_dir or default_data_dir(),
        output_path=out,
        run_cv=not args.no_cv,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
