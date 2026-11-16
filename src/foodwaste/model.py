"""Features, the forecasting model, and the staff-style baseline it is compared to."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
EVENTS = ["none", "assembly", "exam_day", "early_dismissal", "spirit_day"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar, event and lag features. Lags only look backward, so no leakage."""
    df = df.sort_values(["item", "date"]).copy()
    g = df.groupby("item")["demand"]
    df["lag_1"] = g.shift(1)
    df["lag_5"] = g.shift(5)  # same weekday last week
    df["roll_mean_10"] = g.transform(lambda s: s.shift(1).rolling(10).mean())
    df["roll_std_10"] = g.transform(lambda s: s.shift(1).rolling(10).std())
    for d in DOW:
        df[f"dow_{d[:3].lower()}"] = (df["day_of_week"] == d).astype(int)
    for e in EVENTS[1:]:
        df[f"event_{e}"] = (df["event"] == e).astype(int)
    return df.dropna().reset_index(drop=True)


FEATURES = (["attendance_rate", "lag_1", "lag_5", "roll_mean_10", "roll_std_10"]
            + [f"dow_{d[:3].lower()}" for d in DOW]
            + [f"event_{e}" for e in EVENTS[1:]])


def staff_baseline(df: pd.DataFrame, buffer: float = 0.15) -> pd.Series:
    """How production is planned today: recent average plus a safety buffer.

    Staff told us they estimate from what they saw recently and make extra so
    they never run out. Events are not planned for.
    """
    return np.ceil(df["roll_mean_10"] * (1 + buffer))


ITEMS = ["pizza", "burger", "fries", "butter_pasta", "red_sauce_pasta"]
POOLED_FEATURES = FEATURES + [f"item_{i}" for i in ITEMS]


class DemandForecaster:
    """A single gradient-boosted model shared across menu items.

    Instead of predicting raw servings, it predicts demand relative to the
    item's recent average (a ratio near 1.0). That puts pizza and pasta on the
    same scale, so one model learns weekday and event effects from every item's
    history at once: about five times the data a per-item model would get.

    It predicts a high quantile (default 85th percentile) rather than the mean,
    because running out of food is worse than a little waste. The quantile is
    the dial between waste and shortages.
    """

    def __init__(self, quantile: float = 0.85, random_state: int = 0):
        self.quantile = quantile
        self.random_state = random_state
        self.model: GradientBoostingRegressor | None = None

    @staticmethod
    def _X(df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].copy()
        for col in ["lag_1", "lag_5", "roll_std_10"]:
            X[col] = df[col] / df["roll_mean_10"]
        for i in ITEMS:
            X[f"item_{i}"] = (df["item"] == i).astype(int)
        return X[POOLED_FEATURES]

    def fit(self, df: pd.DataFrame) -> "DemandForecaster":
        self.model = GradientBoostingRegressor(
            loss="quantile", alpha=self.quantile, n_estimators=150,
            max_depth=3, learning_rate=0.05, subsample=0.8,
            random_state=self.random_state)
        self.model.fit(self._X(df), df["demand"] / df["roll_mean_10"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        ratio = self.model.predict(self._X(df))
        return np.ceil((ratio * df["roll_mean_10"]).clip(lower=0))
