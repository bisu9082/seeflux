# Daily PARMA W-index from the solar table distributed with PARMA (input/FFPtable.day).
# Layout: header "first_year last_year year1 year2 ..."; rows "month day W(year1) W(year2) ...";
# W >= 10000 flags a ground-level-enhancement (GLE) day (W = value - 10000); negative values are missing.
import datetime as dt, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TABLE = ROOT / "parma" / "cpp" / "input" / "FFPtable.day"


def daily_w(start=dt.date(2009, 1, 1), end=dt.date(2026, 3, 26)):
    lines = TABLE.read_text().split("\n")
    years = [int(x) for x in lines[0].split()[2:]]
    rec = []
    for line in lines[1:]:
        v = line.split()
        if not v:
            continue
        m, d = int(v[0]), int(v[1])
        for y, w in zip(years, v[2:]):
            w = float(w)
            if w < 0:
                continue
            try:
                day = dt.date(y, m, d)
            except ValueError:
                continue
            if start <= day <= end:
                rec.append((pd.Timestamp(day), w % 10000 if w >= 10000 else w, w >= 10000))
    return pd.DataFrame(rec, columns=["date", "W", "gle"]).sort_values("date").reset_index(drop=True)


def annual_mean(year, D=None):
    D = daily_w() if D is None else D
    s = D[(D.date.dt.year == year) & ~D.gle]
    return float(s.W.mean())


def monthly_mean(D=None):
    D = daily_w() if D is None else D
    s = D[~D.gle].copy()
    s["month"] = s.date.dt.to_period("M")
    return s.groupby("month").W.mean().rename("W").reset_index()


if __name__ == "__main__":
    D = daily_w()
    print("days", len(D), "GLE days", int(D.gle.sum()), "P95 W %.1f" % np.percentile(D.W, 95))
    for y in (2019, 2024, 2025):
        print(y, "annual mean W (GLE days excluded) %.1f" % annual_mean(y, D))
