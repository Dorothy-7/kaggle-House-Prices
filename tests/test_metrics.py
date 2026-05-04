from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from house_prices.metrics import rmse_log_space, rmsle_sale_price, saleprice_skew_raw


def test_rmsle_zero_when_predictions_match():
    y = np.array([150_000.0, 185_000.0, 90_000.0])
    assert rmsle_sale_price(y, y) < 1e-12


def test_rmse_log_space_matches_sqrt_mse():
    y_t = np.log1p(np.array([1.0, 2.0, 3.0]))
    y_p = np.log1p(np.array([1.1, 1.9, 3.2]))
    err = rmse_log_space(y_t, y_p)
    expected = float(np.sqrt(mean_squared_error(y_t, y_p)))
    assert abs(err - expected) < 1e-12


def test_saleprice_positive_skew():
    rng = np.random.default_rng(0)
    prices = rng.lognormal(mean=11.8, sigma=0.35, size=300)
    assert saleprice_skew_raw(prices) > 1.0
