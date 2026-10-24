# Fig. 5 (second revision, 2026-10-16): routes at cruise.
# (a) seven great-circle routes over the PARMA vertical cutoff rigidity map, with Natural Earth coastlines;
# (b) ICN-JFK at FL370 at the 2009 annual-mean W (lowest of 2009-2025): PARMA neutrons, neutrons + protons above 10 MeV,
#     the scaled IEC curve and the flat 5600 and 6000 values;
# (c) monthly route fluence (>= 6 km, FL370) of the scaled IEC model divided by PARMA neutrons, 2009 to 2026.
import sys, json, pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from figstyle import DOUBLE, DATA_CMAP, BAND, MODEL, panel_label

P = pd.read_csv(ROOT / "data/step4/routes_levels_profiles_FL370.csv")
M = pd.read_csv(ROOT / "data/step4/rc_map_parma.csv")
MO = pd.concat([pd.read_csv(ROOT / "data/step4/routes_monthly.csv")[["route", "month", "S1_IEC_over_PARMA"]],
                pd.read_csv(ROOT / "data/step4/routes_monthly_south.csv").rename(columns={"IEC_over_PARMA_n": "S1_IEC_over_PARMA"})[
                    ["route", "month", "S1_IEC_over_PARMA"]]])
ROUTES = ["ICN-JFK", "ICN-FRA", "ICN-LAX", "ICN-SYD", "ICN-SIN", "SYD-SCL", "GRU-JNB"]
RCOL = {"ICN-JFK": "#08306b", "ICN-FRA": "#2171b5", "ICN-LAX": "#6a51a3", "ICN-SYD": "#238b45", "ICN-SIN": "#8c6d31",
        "SYD-SCL": "#009E73", "GRU-JNB": "#D55E00"}
LAB = {r: (r.split("-")[1] if r.startswith("ICN") else r.replace("-", "–")) for r in ROUTES}

fig = plt.figure(figsize=(DOUBLE, 3.15), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.05])
ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

# (a) cutoff map, longitude 0-360 so that Pacific routes are continuous
M["lon360"] = M.lon % 360
Z = M.pivot_table(index="lat", columns="lon360", values="Rc").sort_index()
Z[360.0] = Z[0.0]
cf = ax1.contourf(Z.columns, Z.index, Z.values, levels=np.arange(0, 19, 2), cmap=DATA_CMAP, alpha=0.55)
cs = ax1.contour(Z.columns, Z.index, Z.values, levels=[2, 6, 10, 14], colors="white", linewidths=0.5)
ax1.clabel(cs, fmt="%d GV", fontsize=5.5, inline_spacing=2)
for f in json.load(open(ROOT / "data/geo/ne_110m_coastline.geojson", encoding="utf-8"))["features"]:
    xy = np.array(f["geometry"]["coordinates"]); x = xy[:, 0] % 360; y = xy[:, 1]
    cut = np.where(np.abs(np.diff(x)) > 180)[0] + 1
    for xs, ys in zip(np.split(x, cut), np.split(y, cut)):
        ax1.plot(xs, ys, color="#3d3d3d", lw=0.35, zorder=1.5)
for r in ROUTES:
    p = P[(P.route == r) & (P.cond == "y2009")]
    x = p.lon.values % 360; y = p.lat.values
    cut = np.where(np.abs(np.diff(x)) > 180)[0] + 1
    for xs, ys in zip(np.split(x, cut), np.split(y, cut)):
        ax1.plot(xs, ys, color=RCOL[r], lw=1.3)
    if r.startswith("ICN"):
        ax1.text(x[-1], y[-1], " " + LAB[r], color=RCOL[r], fontsize=6.3, fontweight="bold", va="center")
ax1.text(222, -55, "SYD–SCL", color=RCOL["SYD-SCL"], fontsize=6.3, fontweight="bold", ha="center")
ax1.text(350, -16, "GRU–JNB", color=RCOL["GRU-JNB"], fontsize=6.3, fontweight="bold", ha="right")
ax1.plot([126.44], [37.46], marker="*", ms=7, color="#D55E00", mec="white", mew=0.4, zorder=5)
ax1.text(126.44, 31.5, "ICN", fontsize=6.5, ha="center", va="top", color="#D55E00", fontweight="bold")
ax1.set_xlim(0, 360); ax1.set_ylim(-78, 85)
ax1.set_xticks([0, 60, 120, 180, 240, 300, 360]); ax1.set_xticklabels(["0", "60E", "120E", "180", "120W", "60W", "0"])
ax1.set_yticks([-60, -30, 0, 30, 60]); ax1.set_yticklabels(["60S", "30S", "0", "30N", "60N"])
ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
cb = fig.colorbar(cf, ax=ax1, orientation="horizontal", pad=0.02, aspect=35, shrink=0.8, location="top")
cb.set_label("Vertical cutoff rigidity $R_c$ (GV)", fontsize=7); cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=6.5)
panel_label(ax1, "(a)", dy=-0.22)

# (b) ICN-JFK at FL370, 2009 annual-mean W
p = P[(P.route == "ICN-JFK") & (P.cond == "y2009")]
th = p.t_min / 60
ax2.plot(th, p.PARMA_n + p.PARMA_p, color="#08306b", lw=1.3, label="PARMA, neutrons + protons")
ax2.plot(th, p.PARMA_n, color="#2171b5", lw=1.3, ls="--", label="PARMA, neutrons")
ax2.plot(th, p.S1_IEC, color=MODEL["S1_IEC"][0], lw=1.3, label="IEC 62396-1 curves (5600)")
ax2.axhline(6000, color="#444444", lw=0.9, ls="-."); ax2.axhline(5600, color="#999999", lw=0.8, ls=":")
ax2.text(0.3, 6080, "flat 6000", fontsize=6, color="#444444", va="bottom")
ax2.text(0.3, 5520, "flat 5600", fontsize=6, color="#777777", va="top")
ax2.set_xlabel("Flight time (h)"); ax2.set_ylabel("Flux > 10 MeV (cm$^{-2}$ h$^{-1}$)")
ax2.set_xlim(0, th.max()); ax2.set_ylim(0, 9000)
ax2.legend(loc="lower center", fontsize=5.6, handlelength=2.0, borderaxespad=0.2)
ax2.text(0.03, 0.97, "ICN–JFK, FL370, 2009 ($W$=6.6)", transform=ax2.transAxes, va="top", fontsize=6.5)
panel_label(ax2, "(b)", dy=-0.22)

# (c) monthly scaled IEC / PARMA route fluence
ax3.axhspan(1 / 1.3, 1.3, color=BAND, lw=0, zorder=0)
ax3.axhline(1.0, color="#444444", lw=0.7)
for r in ROUTES:
    s = MO[MO.route == r]
    ax3.plot(pd.PeriodIndex(s.month, freq="M").to_timestamp(), s.S1_IEC_over_PARMA, color=RCOL[r], lw=0.9, label=LAB[r])
ax3.set_ylabel("Route fluence, IEC curves / PARMA"); ax3.set_xlabel("Month")
ax3.set_ylim(0.75, 2.6); ax3.set_yticks([0.8, 1.0, 1.3, 1.6, 2.0, 2.4])
ax3.xaxis.set_major_locator(mdates.YearLocator(4)); ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.legend(loc="upper center", fontsize=5.3, ncol=4, handlelength=1.1, columnspacing=0.5, borderaxespad=0.2)
panel_label(ax3, "(c)", dy=-0.22)

out = HERE / "fig_05_routes"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
print("saved")
