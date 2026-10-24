# Generate LaTeX tables directly from the analysis outputs (no hand-typed numbers).
# Revised after the Step 8 review: Table I adds the flat design values, a date-matched JESD89 variant and the share
# below one at the reference-scale bounds k = 0.88, 1 and 1.37; Table II uses the annual-mean solar conditions.
import json, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
J = json.load(open(ROOT / "data/experiment_summary.json"))
OUT = ROOT / "latex/tables"; OUT.mkdir(parents=True, exist_ok=True)
NAME = {"S1_IEC": "IEC 62396-1 curves, 5600", "S1_IEC6000": "IEC 62396-1 curves, 6000", "S1_IEC_alt45": "IEC 62396-1 altitude curve, 45$^{\\circ}$", "FLAT5600": "Flat 5600",
        "FLAT6000": "Flat 6000", "S2_JESD_mid": "JESD89, mid-level solar", "S2_JESD_lin": "JESD89, date-matched solar$^{a}$",
        "S4_AN_ABB": "AN4589, ABB", "S4b_AN_CoolCAD": "AN4589, CoolCAD", "FORMULA": "Surrogate, (1)"}
pct = lambda x: ("%.1f" % (100 * x)) if 0 < 100 * x < 1 else ("%d" % int(100 * x + 0.5))   # half up; one decimal below 1%
f2 = lambda x: "%.2f" % x

# Table I: audit on the independent set, cruise 9-12.5 km
T = pd.read_csv(ROOT / "data/step4/revision_k_flat_jesd.csv").set_index("model")
rows = []
for m in ["S1_IEC", "S1_IEC6000", "S1_IEC_alt45", "FLAT5600", "FLAT6000", "S2_JESD_mid", "S2_JESD_lin", "S4_AN_ABB", "S4b_AN_CoolCAD", "FORMULA"]:
    v = T.loc[m]
    rows.append(f"{NAME[m]} & $n$ & {f2(v['median'])} & {f2(v.p05)}--{f2(v.p95)} & {pct(v['out30_k1.0'])} & {pct(v.out2)} & "
                f"{pct(v['below1_k0.88'])} & {pct(v['below1_k1.0'])} & {pct(v['below1_k1.37'])} & {f2(v.spread)} \\\\")
# Step 8 round 9 (F2): the 6000 floor on the neutron basis the memorandum states, in the n block
import numpy as _np
_Vn = pd.read_csv(ROOT / "data/step4/validation_direct.csv"); _Cn = _Vn[_Vn.alt_km.between(9.0, 12.5)]
_rn = (_np.maximum(6000.0, _Cn.S1_IEC * 6000 / 5600) / _Cn.PARMA).values
_a, _b = _np.percentile(_rn, [5, 95])
rows.insert(5, f"Larger of 6000 and curves & $n$ & {f2(_np.median(_rn))} & {f2(_a)}--{f2(_b)} & {pct(_np.mean((_rn > 1.3) | (_rn < 1 / 1.3)))} & "
               f"{pct(_np.mean((_rn > 2) | (_rn < 0.5)))} & {pct(_np.mean(0.88 * _rn < 1))} & {pct(_np.mean(_rn < 1))} & "
               f"{pct(_np.mean(1.37 * _rn < 1))} & {f2(_b / _a)} \\\\")
# Step 8 round 8 (M3): the 6000 value against neutrons + protons, flat, scaled, and as a floor under the scaled curves
import numpy as _np
_V = pd.read_csv(ROOT / "data/step4/validation_direct.csv"); _P = pd.read_csv(ROOT / "data/step4/r7_points_np.csv")
_V["np"] = _P.PARMA_n2 + _P.PARMA_p
_C = _V[_V.alt_km.between(9.0, 12.5)]
_i6 = _C.S1_IEC * 6000 / 5600
rows.append(r"\midrule")
for lab, num in (("IEC 62396-1 curves, 6000", _i6), ("Flat 6000", _np.full(len(_C), 6000.0)),
                 ("Larger of 6000 and curves", _np.maximum(6000.0, _i6))):
    r = (num / _C.np).values
    q05, q95 = _np.percentile(r, [5, 95])
    rows.append(f"{lab} & $n+p$ & {f2(_np.median(r))} & {f2(q05)}--{f2(q95)} & {pct(_np.mean((r > 1.3) | (r < 1 / 1.3)))} & "
                f"{pct(_np.mean((r > 2) | (r < 0.5)))} & {pct(_np.mean(0.88 * r < 1))} & {pct(_np.mean(r < 1))} & "
                f"{pct(_np.mean(1.37 * r < 1))} & {f2(q95 / q05)} \\\\")
t1 = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{4pt}
\caption{Simplified models against PARMA at cruise altitude (9 to 12.5 km), independent set (802 points; 800 for the surrogate)}
\label{tab:audit}
\begin{tabular}{llcccccccc}
\toprule
 & & & & & & \multicolumn{3}{c}{$k\,r<1$ (\%)} & \\
\cmidrule(lr){7-9}
Model & Ref. & Median $r$ & $r$, 5th--95th pct. & Outside $\pm$30\% (\%) & Outside $\times$2 (\%) & $k=0.88$ & $k=1$ & $k=1.37$ & P95/P5 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip\footnotesize $r$ = model/PARMA, against PARMA neutrons ($n$) or neutrons and protons ($n+p$) above 10~MeV; $k$ = assumed ratio of PARMA to the true flux (Section~\ref{sec:reference}).
$^{a}$Adapted from JESD89B Annex A.3: BSYD branches interpolated linearly in $W$ between the lowest annual mean outside 2008 to 2010 and the highest annual mean of 2009 to 2025, end points taken from the test window.
\end{table*}
"""
(OUT / "tab_audit.tex").write_text(t1, encoding="utf-8")

# Table II: flat values and scaled curves on seven routes (routes_levels.py)
L = pd.read_csv(ROOT / "data/step4/routes_levels.csv").set_index(["route", "level", "cond"])
rows = []
PF = pd.read_csv(ROOT / "data/step4/routes_levels_profiles_FL370.csv")
PF = PF[(PF.cond == "y2009")]
LA = pd.read_csv(ROOT / "data/step4/life_average.csv").set_index(["route", "level"])
for rt in ["ICN-JFK", "ICN-FRA", "SYD-SCL", "ICN-LAX", "ICN-SYD", "ICN-SIN", "GRU-JNB"]:
    a, b, c = L.loc[(rt, "FL370", "y2009")], L.loc[(rt, "FL370", "y2025")], L.loc[(rt, "FL410", "y2009")]
    pf = PF[(PF.route == rt) & (PF.alt_km >= PF[PF.route == rt].alt_km.max() - 1e-9)]
    la3, la4 = LA.loc[(rt, "FL370")], LA.loc[(rt, "FL410")]   # Solar Cycle 24 cruise means (life_average.py), Step 8 round 8
    rows.append(f"{rt.replace('-', '--')} & {a.Rc_min:.1f} & {f2(a.IEC_over_PARMA_n)} & {f2(b.IEC_over_PARMA_n)} & "
                f"{la3.cruise_mean_np_cycle24:.0f} & {la4.cruise_mean_np_cycle24:.0f} & {a.max_np_cruise:.0f} & {c.max_np_cruise:.0f} & "
                f"{pct(a.frac_cruise_np_gt6000)} & {pct(c.frac_cruise_np_gt6000)} \\\\")
tr = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3.5pt}
\caption{Scaled IEC curves and the flat design value on seven routes: solar-minimum (2009 annual mean, $W = 6.6$) and solar-maximum (2025, $W = 107.0$) conditions, and cruise means over Solar Cycle 24}
\label{tab:routes}
\begin{tabular}{lccccccccc}
\toprule
 & & \multicolumn{2}{c}{IEC curves / PARMA, FL370} & \multicolumn{4}{c}{PARMA $n+p$ (cm$^{-2}$\,h$^{-1}$)} & \multicolumn{2}{c}{Cruise time $n+p>6000$ (\%), 2009} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){9-10}
Route & $R_{c,\min}$ (GV) & 2009 & 2025 & \shortstack{Cycle 24\\mean, FL370} & \shortstack{Cycle 24\\mean, FL410} & \shortstack{2009 peak\\FL370} & \shortstack{2009 peak\\FL410} & FL370 & FL410 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip\footnotesize Route fluence above 6~km; IEC curves scale the 5600 neutron value and are compared with PARMA neutrons; $n+p$: PARMA neutrons and protons above 10~MeV.
\end{table*}
"""
(OUT / "tab_routes.tex").write_text(tr, encoding="utf-8")

# Table III: device-level upset counts on seven routes at FL370 (code/device_r7.py, Step 8 round 7): neutron + proton
# references for the saturating (M2) and GeV (M3) fits, and the IEC count on its 6000 basis with three single cross
# sections: sigma(200 MeV), and the M2 or M3 cross section averaged over the sea-level spectrum from 10 to 800 MeV
# (a stand-in for a spallation-beam test); share of the M3 count above 1 GeV.
S = pd.read_csv(ROOT / "data/step4/device_r7.csv")
S = S[S.level == "FL370"]
lab = {"y2019": "2019 (min)", "y2025": "2025 (max)"}
order = ["ICN-JFK", "ICN-FRA", "SYD-SCL", "ICN-LAX", "GRU-JNB", "ICN-SYD", "ICN-SIN"]
rows = []
for rt in order:
    for cond in ("y2019", "y2025"):
        r = S[(S.route == rt) & (S.cond == cond)].iloc[0]
        gev = r.M3_band_1000_3000 + r.M3_band_3000_100000
        rows.append(f"{rt.replace('-', '--')} & {lab[cond]} & {r.ref_M2_np * 1e6:.2f} & {r.ref_M3_np * 1e6:.1f} & "
                    f"{f2(r.IEC_sigM2_s200_over_M2np)} & {f2(r.IEC_sigM2_sp800_over_M2np)} & {f2(r.IEC_sigM2_s200_over_M3np)} & "
                    f"{f2(r.IEC_sigM3_sp800_over_M3np)} & {f2(r.IEC_sigM3_s230_over_M3np)} & {100 * gev:.0f} \\\\")
t2 = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3.5pt}
\caption{Upsets per flight in one 16-Mbit SRAM at FL370 (annual means): neutron and proton reference counts, M2 and M3 fits, and the IEC count (6000 basis) with single cross sections}
\label{tab:ser}
\begin{tabular}{llcccccccc}
\toprule
Route & Year & \shortstack{$N_{\mathrm{M2}}$\\($10^{-6}$)} & \shortstack{$N_{\mathrm{M3}}$\\($10^{-6}$)} & \shortstack{$\sigma^{\mathrm{M2}}_{200}$\\/ $N_{\mathrm{M2}}$} & \shortstack{$\bar\sigma^{\mathrm{M2}}_{\mathrm{sp}}$\\/ $N_{\mathrm{M2}}$} & \shortstack{$\sigma^{\mathrm{M2}}_{200}$\\/ $N_{\mathrm{M3}}$} & \shortstack{$\bar\sigma^{\mathrm{M3}}_{\mathrm{sp}}$\\/ $N_{\mathrm{M3}}$} & \shortstack{$\sigma^{\mathrm{M3}}_{230}$\\/ $N_{\mathrm{M3}}$} & \shortstack{$N_{\mathrm{M3}}$ above\\1 GeV (\%)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip\footnotesize $N$: neutron plus proton count with the same cross section for both. Ratio columns: IEC count with the cross section named, divided by the reference. $\sigma_{200}$, $\sigma_{230}$: cross section of the fit at 200 and 230~MeV, the highest M2 test energy; $\bar\sigma_{\mathrm{sp}}$: cross section of the fit averaged over the sea-level spectrum from 10 to 800~MeV, a stand-in for a spallation-beam test.
\end{table*}
"""
(OUT / "tab_ser.tex").write_text(t2, encoding="utf-8")

# Table IV (Appendix): surrogate coefficients
F = J["formula"]
SYM = {"b0": "b_0", "b1": "b_1", "b2": "b_2", "a": "a", "n": "n", "c": "c", "d": "d", "e": "e"}
rows = [f"${SYM[n]}$ & ${p:.4g}$ & $[{F['param_CI95'][n][0]:.4g},\\ {F['param_CI95'][n][1]:.4g}]$ \\\\"
        for n, p in zip(F["param_names"], F["params"])]
t4 = r"""\begin{table}[!t]
\centering
\caption{Coefficients of Eq. (1) with 95\% bootstrap percentile intervals ($B = 10\,000$)}
\label{tab:coef}
\begin{tabular}{lcc}
\toprule
Coefficient & Value & 95\% interval \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_coef.tex").write_text(t4, encoding="utf-8")
for p in OUT.glob("*.tex"):  # em dash rule: no '---' and no U+2014 in any generated table
    txt = p.read_text(encoding="utf-8")
    assert "---" not in txt and "\u2014" not in txt, f"em dash in {p.name}"
print("tables written:", [p.name for p in OUT.glob("*.tex")])
