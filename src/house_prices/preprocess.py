from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.special import boxcox1p

PathLike = Union[str, Path]


def load_train_test(data_dir: PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    return train, test


def remove_grliv_outliers(train: pd.DataFrame) -> pd.DataFrame:
    mask = ~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))
    return train.loc[mask].copy()


def add_domain_features(df: pd.DataFrame) -> None:
    df["QualGrLiv"] = df["OverallQual"] * df["GrLivArea"]
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["QualTotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    df["AreaPerRoom"] = df["GrLivArea"] / df["TotRmsAbvGrd"].replace(0, 1)
    df["GarageScore"] = df["GarageCars"] * df["GarageArea"]
    df["Log_GrLivArea"] = np.log1p(df["GrLivArea"])
    df["Log_LotArea"] = np.log1p(df["LotArea"])
    df["Log_TotalSF"] = np.log1p(df["TotalSF"])
    df["QualGrLivLog"] = df["OverallQual"] * df["Log_GrLivArea"]
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
    )
    df["IsNew"] = (df["Age"] <= 5).astype(int)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["BsmtScore"] = df["TotalBsmtSF"] * df["HasBsmt"]


def impute(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
    for col in num_cols:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)
    for col in cat_cols:
        train[col] = train[col].fillna("None")
        test[col] = test[col].fillna("None")
    return num_cols, cat_cols


def apply_skew_boxcox(
    train: pd.DataFrame,
    test: pd.DataFrame,
    num_cols: list[str],
    *,
    lambda_: float = 0.15,
    skew_threshold: float = 0.5,
) -> list[tuple[str, float]]:
    transformed: list[tuple[str, float]] = []
    for col in num_cols:
        combined = pd.concat([train[col], test[col]], axis=0)
        sk = float(combined.skew())
        if abs(sk) > skew_threshold:
            transformed.append((col, sk))
            train[col] = boxcox1p(train[col], lambda_)
            test[col] = boxcox1p(test[col], lambda_)
    return transformed


def one_hot_combine(
    train: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["is_train"] = 1
    test["is_train"] = 0
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    encoded = pd.get_dummies(
        combined, columns=cat_cols, drop_first=True, dtype=int
    )
    tr = encoded[encoded["is_train"] == 1].drop("is_train", axis=1)
    te = encoded[encoded["is_train"] == 0].drop("is_train", axis=1)
    return tr, te


def build_matrices(
    data_dir: PathLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[str, float]]]:
    train, test = load_train_test(data_dir)
    test_id = test["Id"].values

    train = remove_grliv_outliers(train)
    y_train = np.log1p(train["SalePrice"].values)

    train = train.drop(columns=["Id", "SalePrice"])
    test = test.drop(columns=["Id"])

    for df in (train, test):
        add_domain_features(df)

    num_cols, cat_cols = impute(train, test)
    skew_info = apply_skew_boxcox(train, test, num_cols)

    train_e, test_e = one_hot_combine(train, test, cat_cols)
    return (
        train_e.values.astype(np.float64),
        test_e.values.astype(np.float64),
        y_train.astype(np.float64),
        test_id,
        skew_info,
    )
