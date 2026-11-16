"""Walk-forward backtest and the paired t-test from the original study."""
from __future__ import annotations

import pandas as pd
from scipy import stats

from .model import DemandForecaster, staff_baseline


def outcomes(produced: pd.Series, demand: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "produced": produced,
        "served": pd.concat([produced, demand], axis=1).min(axis=1),
        "waste": (produced - demand).clip(lower=0),
        "shortfall": (demand - produced).clip(lower=0),
    })


def walk_forward(df: pd.DataFrame, train_days: int = 60, step_days: int = 5,
                 quantile: float = 0.85) -> pd.DataFrame:
    """Retrain every week on all data so far, then plan the next week.

    This mirrors how a cafeteria would really use it: the model never sees the
    week it is planning.
    """
    dates = sorted(df["date"].unique())
    pieces = []
    for start in range(train_days, len(dates), step_days):
        train = df[df["date"] < dates[start]]
        test = df[df["date"].isin(dates[start:start + step_days])].copy()
        model = DemandForecaster(quantile=quantile).fit(train)
        test["model_plan"] = model.predict(test)
        test["staff_plan"] = staff_baseline(test)
        pieces.append(test)
    return pd.concat(pieces, ignore_index=True)


def summarize(bt: pd.DataFrame) -> dict:
    staff = outcomes(bt["staff_plan"], bt["demand"])
    model = outcomes(bt["model_plan"], bt["demand"])
    daily = pd.DataFrame({"date": bt["date"], "staff": staff["waste"],
                          "model": model["waste"]}).groupby("date").sum()
    t, p = stats.ttest_rel(daily["staff"], daily["model"])

    def block(o):
        return {
            "total_waste_servings": int(o["waste"].sum()),
            "waste_pct_of_produced": round(100 * o["waste"].sum() / o["produced"].sum(), 1),
            "days_item_ran_out_pct": round(100 * (o["shortfall"] > 0).mean(), 1),
            "demand_met_pct": round(100 * o["served"].sum() / bt["demand"].sum(), 1),
        }

    return {
        "test_days": int(bt["date"].nunique()),
        "staff_baseline": block(staff),
        "model": block(model),
        "waste_reduction_pct": round(100 * (1 - model["waste"].sum() / staff["waste"].sum()), 1),
        "paired_t_test": {"t": round(float(t), 2), "p_value": float(f"{p:.2e}")},
    }


def by_item(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item, part in bt.groupby("item"):
        s = outcomes(part["staff_plan"], part["demand"])
        m = outcomes(part["model_plan"], part["demand"])
        rows.append({"item": item,
                     "staff_waste": int(s["waste"].sum()),
                     "model_waste": int(m["waste"].sum()),
                     "reduction_pct": round(100 * (1 - m["waste"].sum() / s["waste"].sum()), 1),
                     "model_ran_out_pct": round(100 * (m["shortfall"] > 0).mean(), 1)})
    return pd.DataFrame(rows).sort_values("reduction_pct", ascending=False)
