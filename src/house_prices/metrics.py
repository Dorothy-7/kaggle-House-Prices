from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse_log_space(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> float:
    """学習時の `y = log1p(SalePrice)` 上での RMSE。

    Kaggle の RMSLE と同じログ差の二乗平均平方根（両方 log1p 空間）になる。
    """
    y_log_true = np.asarray(y_log_true, dtype=np.float64)
    y_log_pred = np.asarray(y_log_pred, dtype=np.float64)
    return float(np.sqrt(mean_squared_error(y_log_true, y_log_pred)))


def rmsle_sale_price(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """価格オリジナル尺度の `SalePrice` に対する RMSLE（提出・LB と同定義）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_t = np.log1p(np.clip(y_true, 0, None))
    y_p = np.log1p(np.clip(y_pred, 0, None))
    return rmse_log_space(y_t, y_p)


def numeric_skewness(series: pd.Series | np.ndarray) -> float:
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    return float(s.skew())


def saleprice_skew_raw(train_sale_price: pd.Series | np.ndarray) -> float:
    """学習データ `SalePrice` 素の歪度（Box-Cox 採用議論用）。"""
    return numeric_skewness(train_sale_price)

