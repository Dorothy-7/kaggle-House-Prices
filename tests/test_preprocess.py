from __future__ import annotations

import numpy as np
import pandas as pd

from house_prices.preprocess import apply_skew_boxcox, remove_grliv_outliers


def test_remove_grliv_outliers_drops_classic_two_story_outliers():
    train = pd.DataFrame(
        {
            "GrLivArea": [5000, 4500, 2000],
            "SalePrice": [100000, 350000, 200000],
        }
    )
    cleaned = remove_grliv_outliers(train)
    assert len(cleaned) == 2
    assert not ((cleaned["GrLivArea"] == 5000) & (cleaned["SalePrice"] == 100_000)).any()


def test_apply_skew_boxcox_reports_and_transforms_only_high_skew_numeric_cols():
    rng = np.random.default_rng(42)
    long_tail = rng.exponential(scale=5.0, size=120)
    flat = rng.uniform(low=99.0, high=101.0, size=120)
    train = pd.DataFrame({"skewed": long_tail[:80], "flat": flat[:80]})
    test = pd.DataFrame({"skewed": long_tail[80:], "flat": flat[80:]})
    num_cols = ["skewed", "flat"]
    info = apply_skew_boxcox(train, test, num_cols, skew_threshold=0.5)
    names = {c for c, _ in info}
    assert "skewed" in names
    assert "flat" not in names
