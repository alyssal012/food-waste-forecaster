import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from foodwaste.evaluate import outcomes, summarize, walk_forward  # noqa: E402
from foodwaste.model import FEATURES, DemandForecaster, add_features  # noqa: E402
from foodwaste.simulate import ITEMS, simulate  # noqa: E402


def test_simulation_is_reproducible_and_complete():
    a, b = simulate(seed=1, n_days=30), simulate(seed=1, n_days=30)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 30 * len(ITEMS)
    assert (a["demand"] >= 0).all()


def test_lag_features_do_not_leak_same_day_demand():
    df = add_features(simulate(n_days=40))
    raw = simulate(n_days=40).sort_values(["item", "date"])
    pizza = raw[raw["item"] == "pizza"].reset_index(drop=True)
    row = df[df["item"] == "pizza"].iloc[0]
    idx = pizza.index[pizza["date"] == row["date"]][0]
    assert row["lag_1"] == pizza.loc[idx - 1, "demand"]
    assert not df[FEATURES].isna().any().any()


def test_outcomes_math():
    o = outcomes(pd.Series([10, 5]), pd.Series([7, 9]))
    assert o["waste"].tolist() == [3, 0]
    assert o["shortfall"].tolist() == [0, 4]
    assert o["served"].tolist() == [7, 5]


def test_higher_quantile_means_fewer_shortages():
    df = add_features(simulate(n_days=60))
    train, test = df[df["date"] < "2025-11-01"], df[df["date"] >= "2025-11-01"]
    low = DemandForecaster(quantile=0.5).fit(train).predict(test)
    high = DemandForecaster(quantile=0.95).fit(train).predict(test)
    assert (test["demand"] > high).mean() < (test["demand"] > low).mean()


def test_backtest_summary_shape():
    df = add_features(simulate(n_days=50))
    s = summarize(walk_forward(df, train_days=30, step_days=10))
    assert s["test_days"] > 0
    assert {"staff_baseline", "model", "paired_t_test"} <= s.keys()
