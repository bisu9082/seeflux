# seeflux: end-to-end analysis for
# "Auditing the Boeing-Model Neutron Flux Procedure Referenced by IEC 62396-1 at Cruise Altitude: Cutoff Rigidity,
#  Solar Phase and Device Response"
#
# Runs every step in STEPS in order and writes data/experiment_summary.json. All reported numbers are read from the step
# outputs in data/step4; nothing is typed in by hand. Core steps: Boeing curves digitized from Normand 1996; PARMA 4.10
# reference grid; audit of the simplified models on a geographic grid and on 2000 random points; surrogate fit; ER-2 and
# NYC benchmarks; reference-scale envelope; routes, flight levels and service-life means; device folds; robustness checks
# added in review (CARI-7 cross-check, break-even scales, cutoff epoch, paired bootstrap, counterfactual substitution,
# subsets, joint sensitivity, a fresh confirmation sample); build check of PARMA.
# Requirements: Python packages in requirements.txt; PARMA 4.10 C++ compiled in WSL Ubuntu (see README_code.md);
# the FAA CARI-7 distribution (version of 15 November 2018, Windows executable) for the CARI-7 steps.
import subprocess, sys, json, time, pathlib, shutil
import pandas as pd

CODE = pathlib.Path(__file__).resolve().parent
ROOT = CODE.parent
OUT = ROOT / "data/step4"
STEPS = ["digitize_normand.py", "grid_parma.py", "audit.py", "fit_formula.py", "validate_direct.py",
         "bench_goldhagen.py", "er2_gt10_anchor.py", "er2_geometry.py", "uncertainty_band.py", "audit_revision.py", "audit_revision2.py", "check_bsyd_tableA1.py", "routes.py", "aircraft_sensitivity.py", "ser_routes.py", "routes_levels.py", "cari_crosscheck.py", "k_breakeven.py", "life_average.py", "device_parametric.py", "cutoff_epoch.py", "paired_bootstrap.py", "r7_checks.py", "subsets.py", "counterfactual.py", "device_r7.py", "device_r8.py", "joint_sensitivity.py", "fresh_sample.py", "stored_values.py", "verify_parma.py"]


def run_all(log):
    for s in STEPS:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(CODE / s)], capture_output=True, text=True, cwd=str(ROOT))
        log.write(f"===== {s} ({time.time() - t0:.0f} s, exit {r.returncode})\n{r.stdout}\n{r.stderr}\n")
        if r.returncode != 0:
            raise SystemExit(f"{s} failed, see log")


def summarize():
    au = pd.read_csv(OUT / "audit_summary.csv")
    ub = pd.read_csv(OUT / "uncertainty_band.csv")
    fit = json.load(open(OUT / "formula_fit.json"))
    bench = pd.read_csv(OUT / "bench_goldhagen2004.csv")
    band = json.load(open(OUT / "reference_band.json"))
    rt = pd.read_csv(OUT / "routes_summary.csv")
    ac = pd.read_csv(OUT / "aircraft_sensitivity.csv")
    lat = pd.read_csv(OUT / "latitude_ambiguity_12km_W100.csv")
    grid = pd.read_csv(OUT / "parma_grid.csv")
    sel = fit["selected"]
    ratio_cols = [c for c in rt.columns if c.endswith("_over_PARMA")]
    return {
        "reference_grid": {"n_nodes": int(len(grid)), "W_levels": sorted(grid.W.unique().tolist()),
                           "Rc_range_GV": [float(grid.Rc.min()), float(grid.Rc.max())],
                           "alt_levels_km": sorted(grid.alt_km.unique().tolist())},
        "audit_by_subset": au.to_dict(orient="records"),
        "uncertainty_band": {"k_range": [band["K_LO"], band["K_HI"]], "cruise_9_12p5km": ub.to_dict(orient="records")},
        "latitude_ambiguity_12km_W100": lat.to_dict(orient="records"),
        "formula": {"selected": sel, "param_names": fit["param_names"], "params": fit["candidates"][sel]["params"],
                    "param_CI95": fit["param_CI95"], "training_n": fit["training_nodes"]["n"], "heldout_n": fit["heldout_n"],
                    "heldout": fit["candidates"][sel]["heldout"], "heldout_p95_CI95": fit["heldout_p95_CI95"],
                    "extrapolation": fit["extrapolation"],
                    "heldout_by_group": pd.read_csv(OUT / "formula_heldout_by_group.csv").to_dict(orient="records"),
                    "candidates_AIC_BIC": {k: {"AIC": v["AIC"], "BIC": v["BIC"], "heldout": v["heldout"]}
                                           for k, v in fit["candidates"].items()}},
        "independent_validation_direct_PARMA": pd.read_csv(OUT / "validation_direct_summary.csv").to_dict(orient="records"),
        "benchmark_goldhagen2004": bench.to_dict(orient="records"),
        "benchmark_nyc": json.load(open(OUT / "bench_nyc.json")),
        "routes": rt.to_dict(orient="records"),
        "routes_ratio_range": {c: [float(rt[c].min()), float(rt[c].max())] for c in ratio_cols},
        "ser_demonstration": {"meta": json.load(open(OUT / "ser_meta.json")),
                              "routes": pd.read_csv(OUT / "ser_routes.csv").to_dict(orient="records")},
        "revision_k_flat_jesd": pd.read_csv(OUT / "revision_k_flat_jesd.csv").to_dict(orient="records"),
        "revision_bootstrap_ci": pd.read_csv(OUT / "revision_bootstrap_ci.csv").to_dict(orient="records"),
        "revision_breakdown": pd.read_csv(OUT / "revision_breakdown.csv").to_dict(orient="records"),
        "revision_sensitivity": json.load(open(OUT / "revision_sensitivity.json")),
        "routes_monthly_IEC_over_PARMA": pd.read_csv(OUT / "routes_monthly.csv").groupby("route").S1_IEC_over_PARMA.describe(
            percentiles=[.05, .95]).reset_index().to_dict(orient="records"),
        "parma_verification": json.load(open(OUT / "parma_verification.json")),
        "aircraft_sensitivity_gt10":ac.groupby("g").ratio_gt10.agg(["min", "median", "max"]).reset_index().to_dict(orient="records"),
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--summary-only" not in sys.argv:  # keep the log of the last full run when only summarizing
        with open(ROOT / "data/run_log.txt", "w", encoding="utf-8") as log:
            run_all(log)
    summary = summarize()
    json.dump(summary, open(ROOT / "data/experiment_summary.json", "w"), indent=1)
    shutil.copy(OUT / "parma_grid.csv", ROOT / "data/raw_results.csv")
    print("wrote data/experiment_summary.json and data/raw_results.csv (PARMA reference grid)")
