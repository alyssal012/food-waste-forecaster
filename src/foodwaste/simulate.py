"""Simulated cafeteria data.

The original study used one week of hand-recorded servings, and that
data is no longer available. This module makes a larger, realistic stand-in dataset with
the same menu items and the same demand drivers that came up in the staff
interviews and student survey: day of week, attendance, school events, item
popularity, and seasonal drift.

Everything here is synthetic. Do not read results from it as real-world findings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ITEMS = {
    # item: (base daily demand, day-of-week multipliers Mon..Fri)
    "pizza":            (140, [1.00, 0.95, 1.05, 0.95, 1.25]),
    "burger":           (95,  [1.05, 1.00, 0.95, 1.00, 1.10]),
    "fries":            (120, [1.00, 1.00, 1.00, 1.00, 1.15]),
    "butter_pasta":     (60,  [1.10, 1.00, 1.00, 1.05, 0.85]),
    "red_sauce_pasta":  (55,  [0.95, 1.05, 1.10, 1.00, 0.80]),
}

EVENTS = {
    # event: (probability on a school day, demand multiplier)
    "none":            (0.80, 1.00),
    "assembly":        (0.06, 0.85),
    "exam_day":        (0.07, 0.75),
    "early_dismissal": (0.04, 0.35),
    "spirit_day":      (0.03, 1.15),
}


def school_days(start: str = "2025-09-02", n_days: int = 175) -> pd.DatetimeIndex:
    days = pd.bdate_range(start=start, periods=n_days + 25)
    # drop a winter break and a spring break
    mask = ~(((days >= "2025-12-22") & (days <= "2026-01-02")) |
             ((days >= "2026-03-30") & (days <= "2026-04-03")))
    return days[mask][:n_days]


def simulate(seed: int = 7, n_days: int = 175) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = school_days(n_days=n_days)
    event_names = list(EVENTS)
    event_probs = [EVENTS[e][0] for e in event_names]

    rows = []
    for i, day in enumerate(days):
        event = rng.choice(event_names, p=event_probs)
        attendance = float(np.clip(rng.normal(0.94, 0.025), 0.80, 1.0))
        if day.month in (12, 1, 2):  # flu season
            attendance -= 0.02
        # slow seasonal drift: novelty wears off, then a spring bump
        season = 1.0 - 0.06 * np.sin(np.pi * i / len(days))
        for item, (base, dow) in ITEMS.items():
            mean = (base * dow[day.weekday()] * attendance / 0.94
                    * EVENTS[event][1] * season)
            demand = int(max(0, rng.poisson(mean) + rng.normal(0, 4)))
            rows.append({
                "date": day.date().isoformat(),
                "item": item,
                "day_of_week": day.day_name(),
                "event": event,
                "attendance_rate": round(attendance, 3),
                "demand": demand,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parents[2] / "data" / "cafeteria_demand.csv"
    df = simulate()
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows to {out}")
