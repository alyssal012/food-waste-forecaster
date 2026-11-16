"""Generate data, backtest the model against the staff baseline, write a report."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from foodwaste.evaluate import by_item, summarize, walk_forward  # noqa: E402
from foodwaste.model import add_features  # noqa: E402
from foodwaste.simulate import simulate  # noqa: E402


def main() -> None:
    raw = simulate()
    raw.to_csv(ROOT / "data" / "cafeteria_demand.csv", index=False)
    df = add_features(raw)

    bt = walk_forward(df)
    summary = summarize(bt)
    items = by_item(bt)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "summary.json").write_text(json.dumps(summary, indent=2))
    items.to_csv(reports / "by_item.csv", index=False)

    # Waste over time, weekly totals
    weekly = bt.assign(week=pd.to_datetime(bt["date"]).dt.to_period("W").dt.start_time)
    weekly["staff_waste"] = (weekly["staff_plan"] - weekly["demand"]).clip(lower=0)
    weekly["model_waste"] = (weekly["model_plan"] - weekly["demand"]).clip(lower=0)
    w = weekly.groupby("week")[["staff_waste", "model_waste"]].sum()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(w.index, w["staff_waste"], label="Current practice (recent avg + 15%)", color="#9a9a9a")
    ax.plot(w.index, w["model_waste"], label="Forecast model (85th percentile)", color="#2a7f62")
    ax.set_ylabel("Servings wasted per week")
    ax.set_title("Weekly food waste, walk-forward backtest (simulated data)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(reports / "weekly_waste.png", dpi=150)

    # Waste vs. shortage tradeoff: sweep the quantile dial
    frontier = []
    for q in [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        sq = summarize(walk_forward(df, quantile=q))["model"]
        frontier.append({"quantile": q, "waste_pct": sq["waste_pct_of_produced"],
                         "ran_out_pct": sq["days_item_ran_out_pct"]})
    frontier = pd.DataFrame(frontier)
    frontier.to_csv(reports / "tradeoff.csv", index=False)
    base = summary["staff_baseline"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(frontier["ran_out_pct"], frontier["waste_pct"], "o-", color="#2a7f62",
            label="Forecast model")
    for _, r in frontier.iterrows():
        ax.annotate(f"q={r['quantile']:.2f}", (r["ran_out_pct"], r["waste_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.scatter([base["days_item_ran_out_pct"]], [base["waste_pct_of_produced"]],
               color="#9a9a9a", s=60, zorder=3, label="Current practice")
    ax.set_xlabel("Item-days that ran out (%)")
    ax.set_ylabel("Food wasted (% of servings made)")
    ax.set_title("Waste vs. shortage tradeoff (simulated data)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(reports / "tradeoff.png", dpi=150)

    print(json.dumps(summary, indent=2))
    print(frontier.to_string(index=False))
    print(items.to_string(index=False))


if __name__ == "__main__":
    main()
