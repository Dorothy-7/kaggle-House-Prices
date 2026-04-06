from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """`src/house_prices/config.py` から見たリポジトリ直下 (Python3/)。"""
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    """Kaggle 展開先。リポジトリでは `data/train.csv` を想定。"""
    return project_root() / "data"


def default_output_path() -> Path:
    return project_root() / "outputs" / "submission_boxcox_5models_ensemble.csv"
