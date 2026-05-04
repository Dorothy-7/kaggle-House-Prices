from __future__ import annotations

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 16,
    "learning_rate": 0.03,
    "n_estimators": 15000,
    "min_child_samples": 30,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "random_state": 123,
    "verbose": -1,
}

XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.03,
    "n_estimators": 10000,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "random_state": 123,
    "early_stopping_rounds": 100,
}

ENSEMBLE_WEIGHTS = (0.35, 0.22, 0.18, 0.15, 0.10)  # LGB, XGB, Ridge, Lasso, CatBoost


def cv_lgb_rmse(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 123,
) -> tuple[float, float]:
    params = {**LGB_PARAMS}
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics: list[float] = []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        pred = model.predict(X_val)
        metrics.append(float(np.sqrt(mean_squared_error(y_val, pred))))
    return float(np.mean(metrics)), float(np.std(metrics))


def predict_lgb_mean(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seeds: tuple[int, ...] = (123, 456, 789, 42, 2024),
) -> np.ndarray:
    preds = []
    params = {**LGB_PARAMS}
    for seed in seeds:
        params["random_state"] = seed
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        preds.append(model.predict(X_test))
    return np.mean(preds, axis=0)


def predict_xgb_mean(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seeds: tuple[int, ...] = (123, 456, 789),
) -> np.ndarray:
    preds = []
    for seed in seeds:
        p = {**XGB_PARAMS, "random_state": seed}
        model = xgb.XGBRegressor(**p)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train)],
            verbose=False,
        )
        preds.append(model.predict(X_test))
    return np.mean(preds, axis=0)


def predict_ridge_mean(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alphas: tuple[float, ...] = (10, 15, 20),
) -> np.ndarray:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    preds = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_tr, y_train)
        preds.append(model.predict(X_te))
    return np.mean(preds, axis=0)


def predict_lasso_mean(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alphas: tuple[float, ...] = (0.0005, 0.001, 0.0015),
) -> np.ndarray:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    preds = []
    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_tr, y_train)
        preds.append(model.predict(X_te))
    return np.mean(preds, axis=0)


def predict_catboost_mean(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seeds: tuple[int, ...] = (123, 456, 789),
) -> np.ndarray:
    preds = []
    for seed in seeds:
        model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5,
            random_seed=seed,
            verbose=False,
            early_stopping_rounds=100,
        )
        model.fit(X_train, y_train, eval_set=(X_train, y_train), verbose=False)
        preds.append(model.predict(X_test))
    return np.mean(preds, axis=0)


def blend_log_predictions(
    lgb_p: np.ndarray,
    xgb_p: np.ndarray,
    ridge_p: np.ndarray,
    lasso_p: np.ndarray,
    cat_p: np.ndarray,
    weights: tuple[float, float, float, float, float] = ENSEMBLE_WEIGHTS,
) -> np.ndarray:
    w_lgb, w_xgb, w_ridge, w_lasso, w_cat = weights
    return (
        w_lgb * lgb_p
        + w_xgb * xgb_p
        + w_ridge * ridge_p
        + w_lasso * lasso_p
        + w_cat * cat_p
    )


def predict_sale_prices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    run_cv: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    cv_stats: tuple[float, float] | None = None
    if run_cv:
        mean_rmse, std_rmse = cv_lgb_rmse(X_train, y_train)
        cv_stats = (mean_rmse, std_rmse)
        if verbose:
            print(f"[CV LGB RMSE] {mean_rmse:.5f}±{std_rmse:.5f}")

    if verbose:
        print("LightGBM (5 seeds)...")
    lgb_p = predict_lgb_mean(X_train, y_train, X_test)
    if verbose:
        print("XGBoost (3 seeds)...")
    xgb_p = predict_xgb_mean(X_train, y_train, X_test)
    if verbose:
        print("Ridge (3 alphas)...")
    ridge_p = predict_ridge_mean(X_train, y_train, X_test)
    if verbose:
        print("Lasso (3 alphas)...")
    lasso_p = predict_lasso_mean(X_train, y_train, X_test)
    if verbose:
        print("CatBoost (3 seeds)...")
    cat_p = predict_catboost_mean(X_train, y_train, X_test)

    y_log = blend_log_predictions(lgb_p, xgb_p, ridge_p, lasso_p, cat_p)
    y_price = np.expm1(y_log)
    return y_price, cv_stats
