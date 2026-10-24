# Supplementary tables S1-S3 from data/step_ml outputs (no hand-typed numbers). Ranges as "a to b"; no em dashes.
import pathlib, json
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
ML, TAB = ROOT / "data/step_ml", ROOT / "latex/tables"


def write(name, body):
    assert "—" not in body and "---" not in body, name
    (TAB / name).write_text(body, encoding="utf-8", newline="\n")
    print("wrote", name)


# S1: emulators (P95 and max, %)
E = pd.read_csv(ML / "ml1_emulators.csv")
N = pd.read_csv(ML / "ml4_nested.csv")
tests = [("independent (1995)", "in-domain: independent (1995)", "Independent"),
         ("extrapolate W>=100 (train W<=75)", "W>=100 (train W<=75)", "$W\\geq100$"),
         ("extrapolate Rc>12 (train Rc<=12)", "Rc>12 (train Rc<=12)", "$R_c>12$"),
         ("extrapolate alt>12.5 (train alt<=12)", "alt>12.5 (train alt<=12)", "$h>12.5$"),
         ("extrapolate alt<9 (train alt>=9)", "alt<9 (train alt>=9)", "$h<9$")]
models = ["Eq. (1)", "Ridge, cubic poly", "Gaussian process", "MLP ensemble", "XGBoost", "kNN (k=4, distance)"]
names = {"Eq. (1)": "Eq. (1)", "Ridge, cubic poly": "Ridge, cubic polynomial", "Gaussian process": "Gaussian process",
         "MLP ensemble": "MLP, 5-seed ensemble", "XGBoost": "Gradient-boosted trees", "kNN (k=4, distance)": "$k$-NN, $k=4$"}


def cell(r):
    p = f"{r.p95_pct:.0f}" if r.p95_pct >= 100 else f"{r.p95_pct:.1f}"
    return f"{p} ({r.max_pct:.0f})"


rows = []
for m in models:
    rows.append(names[m] + " & " + " & ".join(cell(E[(E.model == m) & (E.test == a)].iloc[0]) for a, _, _ in tests) + r" \\")
rows.append("Linear, 10 terms$^\\dagger$ & " + " & ".join(cell(N[(N.model == "nested k=10") & (N.test == b)].iloc[0]) for _, b, _ in tests) + r" \\")
write("supp_emulators.tex", "\n".join([
    r"\begin{table*}[t]", r"\centering",
    r"\caption{Emulators trained on the 252 nodes of Eq. (1): 95th percentile (largest) absolute error relative to PARMA, \%}",
    r"\label{tab:s_emul}", r"\begin{tabular}{lccccc}", r"\toprule",
    r"Model & " + " & ".join(t[2] for t in tests) + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
    r"\par\smallskip\footnotesize Columns 2 to 5: region hold-outs, refitted without the region named. $^\dagger$Terms reselected inside each hold-out.",
    r"\end{table*}", ""]))

# S2: R2 of nested input sets and SHAP shares
A = pd.read_csv(ML / "ml2_attribution_r2.csv")
S = pd.read_csv(ML / "ml2_shap_shares.csv").set_index("model")
rows = [f"{r.model} & {r._4:.2f} & {r._5:.2f} & {r._6:.2f} & {r._7:.3f} & "
        + " & ".join(f"{S.loc[r.model, c]:.2f}" for c in ["alt_km", "abslat", "W", "Rc"]) + r" \\"
        for r in A.itertuples()]
write("supp_attribution.tex", "\n".join([
    r"\begin{table*}[t]", r"\centering",
    r"\caption{Held-out $R^2$ of $\ln r$ for nested input sets and mean $|$SHAP$|$ shares, cruise nodes of the geographic grid (70\,560)}",
    r"\label{tab:s_attr}", r"\begin{tabular}{lcccccccc}", r"\toprule",
    r" & \multicolumn{4}{c}{$R^2$, inputs} & \multicolumn{4}{c}{SHAP share} \\",
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
    r"Model & $h,|\lambda|$ & $+W$ & $+R_c$ & $+W+R_c$ & $h$ & $|\lambda|$ & $W$ & $R_c$ \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]))

# S3: rule for IEC ratio below one
C = pd.read_csv(ML / "ml3_noncons_rule.csv")
FEAT = {"A: alt,|lat|,W": r"$h, |\lambda|, W$", "B: alt,Rc,W": r"$h, R_c, W$"}
rows = [f"{FEAT[r.features]} & {r.model} & {r.balanced_acc:.2f} & {r.recall_noncons:.2f} & {r.flagged_share:.2f}" + r" \\"
        for r in C.itertuples()]
write("supp_rule.tex", "\n".join([
    r"\begin{table}[t]", r"\centering",
    r"\caption{Classifiers for IEC ratio below one, trained on the grid and tested on the 802 independent cruise points (base rate " + f"{C.base_rate.iloc[0]:.2f})" + "}",
    r"\label{tab:s_rule}", r"\begin{tabular}{llccc}", r"\toprule",
    r"Inputs & Model & Bal. acc. & Recall & Flagged \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]))

# S4: altitude-dependent bounding environment (audit_revision2.py)
Bd = pd.read_csv(ROOT / "data/step4/revision2_bounding_by_level.csv")
Bd = Bd[Bd.Rc == 0]
import sys as _sys; _sys.path.insert(0, str(ROOT / "code")); import standards as _S
rows = [f"FL{int(r.FL)} & {r.alt_km:.2f} & {r.n:.0f} & {r.np:.0f} & {r.np_over_k088:.0f} & {float(_S.s1_iec_boeing_6000(r.alt_km, 89.0)):.0f}" + r" \\"
        for r in Bd.itertuples()]
write("supp_bounding.tex", "\n".join([
    r"\begin{table}[t]", r"\centering",
    r"\caption{PARMA flux above 10~MeV at $R_c = 0$ (identical up to 1~GV) and $W = 0$ by flight level, n\,cm$^{-2}$\,h$^{-1}$ ($n$: neutrons, $p$: protons)}",
    r"\label{tab:s_bound}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccc}", r"\toprule",
    r"Level & km & $n$ & $n+p$ & $(n+p)/0.88$ & IEC, 6000 at 89$^{\circ}$ \\", r"\midrule", *rows,
    r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]))

# Table SV: break-even reference scale (code/k_breakeven.py), Step 8 round 6 (M2)
KB = pd.read_csv(ROOT / "data/step4/k_breakeven.csv")
LAB = {"peak n+p above 6000 somewhere on the route, FL370 (max over routes: ICN-JFK)": "6000 exceeded at peak, FL370",
       "peak n+p above 6000 somewhere on the route, FL390 (max over routes: ICN-JFK)": "6000 exceeded at peak, FL390",
       "peak n+p above 6000 somewhere on the route, FL410 (max over routes: ICN-JFK)": "6000 exceeded at peak, FL410",
       "cruise-mean n+p above 6000, ICN-JFK FL370 2009": "6000 exceeded on cruise mean, ICN--JFK",
       "cruise-mean n+p above 6000, ICN-FRA FL370 2009": "6000 exceeded on cruise mean, ICN--FRA",
       "cruise-mean n+p above 6000, SYD-SCL FL370 2009": "6000 exceeded on cruise mean, SYD--SCL",
       "below the flux in at least half of the months": "IEC below flux, GRU--JNB, half of months",
       "below the flux in at least one month": "IEC below flux, GRU--JNB, any month",
       "IEC scaled median below the flux": "Scaled IEC median below flux",
       "JESD89 median below the flux": "JESD89 median below flux",
       "AN4589 ABB median below the flux": "AN4589 (ABB) median below flux",
       "AN4589 CoolCAD median below the flux": "AN4589 (CoolCAD) median below flux",
       "FL370 value (7498) at or above 6000": "Bounding value $\\ge$ 6000, FL370",
       "FL410 value (9427) at or above 6000": "Bounding value $\\ge$ 6000, FL410"}
yn = lambda b: "yes" if b else "no"
rows = [f"{LAB[r.finding]} & {r.k_star:.2f} & {yn(r.k_star > 1.0748)} & {yn(r.k_star > 1.2838)} & {yn(r.k_star > 1.37)} \\\\"
        for r in KB.itertuples() if r.finding in LAB]
write("supp_kstar.tex", "\n".join([
    r"\begin{table*}[t]", r"\centering",
    r"\caption{Break-even reference scale $k^{*}$: each finding holds for $k < k^{*}$ ($k$ = PARMA / true flux); solar-minimum years, FL370 unless stated}",
    r"\label{tab:s_kstar}", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{lcccc}", r"\toprule",
    r"Finding & $k^{*}$ & \shortstack{Holds at\\$k = 1.07$} & \shortstack{Holds at\\$k = 1.28$} & \shortstack{Holds at\\$k = 1.37$} \\", r"\midrule", *rows,
    r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]))

# Table SVI: service-life averages (code/life_average.py), Step 8 round 6 (M4)
LA = pd.read_csv(ROOT / "data/step4/life_average.csv").set_index(["route", "level"])
FLT = json.load(open(ROOT / "data/step4/life_average.json"))["fleet_hours_weighted"]
rows = []
for rt in ["ICN-JFK", "ICN-FRA", "SYD-SCL", "ICN-LAX", "GRU-JNB", "ICN-SYD", "ICN-SIN"]:
    a, b = LA.loc[(rt, "FL370")], LA.loc[(rt, "FL410")]
    rows.append(f"{rt.replace('-', '--')} & {a.cruise_mean_np_cycle24:.0f} & {b.cruise_mean_np_cycle24:.0f} & "
                f"{b.flight_mean_np_cycle24:.0f} & {b.k_star_cruise_life_cycle24:.2f} & {a.IEC6000_over_np_cycle24:.2f} \\\\")
for key, lab in (("northern_long_haul", "Northern long-haul"), ("seven_routes", "Seven routes")):
    a, b = FLT[f"FL370_{key}"], FLT[f"FL410_{key}"]
    rows.append(f"{lab} & {a['cruise_mean_np_cycle24']:.0f} & {b['cruise_mean_np_cycle24']:.0f} & {b['flight_mean_np_cycle24']:.0f} & "
                f"{b['cruise_mean_np_cycle24'] / 6000:.2f} &  \\\\")
write("supp_life.tex", "\n".join([
    r"\begin{table}[t]", r"\centering",
    r"\caption{PARMA neutron and proton flux above 10~MeV averaged over Solar Cycle 24 (2009 to 2019), n\,cm$^{-2}$\,h$^{-1}$, and the scaled IEC curves on the 6000 basis relative to it}",
    r"\label{tab:s_life}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lccccc}", r"\toprule",
    r"Route & \shortstack{Cruise\\FL370} & \shortstack{Cruise\\FL410} & \shortstack{Flight\\FL410} & \shortstack{$k^{*}$\\FL410} & \shortstack{IEC$_{6000}$/\\$(n+p)$, FL370} \\",
    r"\midrule", *rows[:7], r"\midrule", *rows[7:], r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]))

# Table SVII: parametric device study (code/device_parametric.py), Step 8 round 6 (M6)
DP = pd.read_csv(ROOT / "data/step4/device_parametric.csv")
rows = []
for r in DP.itertuples():
    lab = (f"Generic, {r.E_th:.0f} MeV, $W$ {r.W:.0f}" if r.kind.startswith("generic") else r.response)
    rows.append(f"{lab} & {r.p_over_n_median:.2f} & {r.method_factor_median:.2f} & {r.IEC6000_over_np_median:.2f} & "
                f"{100 * r.IEC6000_below1:.0f} & {r.IEC6000_spread:.2f} & {r.JESD_spread:.2f} & {r.swing_JFK_FL370_2019_over_2025:.2f} \\\\")
write("supp_device.tex", "\n".join([
    r"\begin{table*}[t]", r"\centering",
    r"\caption{Event counts (upsets; latchups for the two SEL parts) for generic and published Weibull responses at the 802 cruise points (neutrons and protons, proton cross section equal to the neutron one): ratio of proton to neutron counts, method factor, the IEC count on the 6000 basis relative to the reference, and the ICN--JFK swing at FL370 between the 2019 and 2025 annual means}",
    r"\label{tab:s_device}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccccc}", r"\toprule",
    r"Response & $N^{p}/N$ & \shortstack{Method\\factor} & \shortstack{IEC$_{6000}$/ref.\\median} & \shortstack{Below\\one (\%)} & \shortstack{IEC\\P95/P5} & \shortstack{JESD89\\P95/P5} & \shortstack{Swing\\2019/2025} \\",
    r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
    r"\par\smallskip\footnotesize Generic responses: $\sigma_{\mathrm{sat}} = 1$, $s = 1.5$, threshold and width $W$ as listed. Published: Weibull fits of Cecchetto et al. (2019), M2 to 230~MeV, M3 with the GeV rise. Method factor: PARMA flux above 10~MeV times $\sigma(200~\mathrm{MeV})$ over the folded neutron count.",
    r"\end{table*}", ""]))

# Table SIX: subsets and traffic proxy (code/subsets.py), Step 8 round 8 (M4)
SB = pd.read_csv(ROOT / "data/step4/subsets.csv")
MOD = {"IEC curves 5600 / n": "IEC 5600, $n$", "IEC curves 6000 / n+p": "IEC 6000, $n+p$",
       "larger of 6000 and curves / n+p": "6000 as floor, $n+p$", "JESD89 / n": "JESD89, $n$"}
rows = []
for sub, g in SB.groupby("subset", sort=False):
    rows.append(r"\multicolumn{6}{l}{\textit{" + sub.replace("-", "--") + f" ({int(g.n.iloc[0])} points)" + r"}} \\")
    for r in g.itertuples():
        rows.append(f"\\quad {MOD[r.model]} & {r.median:.2f} & {100 * r.below1_k1:.0f} ({100 * r.below1_k1_lo:.0f}--{100 * r.below1_k1_hi:.0f}) & "
                    f"{100 * r.below1_k107:.0f} & {100 * r.below1_k128:.0f} & {r.spread:.2f} \\\\")
write("supp_subsets.tex", "\n".join([
    r"\begin{table}[t]", r"\centering",
    r"\caption{Cruise statistics by flight-level band, in the northern long-haul latitude band and with a traffic-proxy weighting: median ratio, share below one at $k$ = 1 (95\% interval), 1.07 and 1.28 (\%), and P95/P5}",
    r"\label{tab:s_subsets}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize", r"\begin{tabular}{lccccc}", r"\toprule",
    r"Subset and model & Median & $k = 1$ & 1.07 & 1.28 & P95/P5 \\", r"\midrule", *rows,
    r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]))

# Table SX: device sensitivity on the routes at FL370 (code/device_r8.py), Step 8 round 8 (M5)
DR = pd.read_csv(ROOT / "data/step4/device_r8.csv")
COLS = [("IEC_s200_over_ref", "$\\sigma_{200}$"), ("IEC_s230_over_ref", "$\\sigma_{230}$"), ("IEC_sp400_over_ref", "$\\bar\\sigma_{\\mathrm{sp}}$, 400"),
        ("IEC_sp800_over_ref", "$\\bar\\sigma_{\\mathrm{sp}}$, 800"), ("IEC_sp1e5_over_ref", "$\\bar\\sigma_{\\mathrm{sp}}$, no limit")]
rows = []
for resp, g in DR.groupby("response", sort=False):
    rows.append(resp.replace("E_th", "$E_{\\mathrm{th}}$").replace("0.23-3", "0.23--3") + " & " +
                " & ".join(f"{g[c].min():.2f}--{g[c].max():.2f}" for c, _ in COLS) + r" \\")
write("supp_device_sens.tex", "\n".join([
    r"\begin{table*}[t]", r"\centering",
    r"\caption{IEC count on its 6000 basis divided by the neutron and proton reference count, seven routes at FL370, 2019 and 2025 annual means (range over routes and years), for the Renesas fits, M2 with other thresholds and M3 with other GeV shapes, and five test cross sections ($\bar\sigma_{\mathrm{sp}}$: averaged over the sea-level shape from 10~MeV to the upper end in MeV)}",
    r"\label{tab:s_devsens}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccc}", r"\toprule",
    r"Response & " + " & ".join(h for _, h in COLS) + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]))
