# seeflux analysis code

Code for the manuscript *Auditing IEC 62396-1 and JESD89 Neutron Flux Values at Cruise Altitude: Cutoff Rigidity, Solar Phase and Proton Contributions* (target: IEEE Transactions on Nuclear Science).

## What it does
1. Digitizes the Boeing (IEC 62396-1 lineage) altitude and latitude curves from Normand 1996, Figs 2 and 3.
2. Builds a PARMA 4.10 reference grid of the >10 MeV, 1 to 10 MeV and total neutron flux. The grid covers W-index 0 to 150, cutoff rigidity Rc 0 to 18 GV, and altitude 0 km plus 6 to 15 km. All runs use the free atmosphere.
3. Audits these simplified models against PARMA on a 5° × 10° geographic grid:
   - S1: IEC/Boeing
   - S1b: IEC/Boeing with Normand's Eq. 3 latitude fit
   - S2: JESD89/Gordon 2004
   - S4 and S4b: Microchip AN4589 (ABB and CoolCAD forms)
4. Fits three pre-specified formula forms on 252 sparse grid nodes and selects one by BIC. The selected form is tested on the remaining grid nodes, on four region hold-out (extrapolation) tests, and with a nonparametric bootstrap of B = 10,000. The formula is in `seeflux_formula.py` (with domain guards).
4b. Independent validation (`validate_direct.py`): 2,000 random date, place and altitude points (2009 to 2026), each computed with a direct PARMA call rather than through the grid. This is the primary validation set.
5. Benchmarks PARMA and CARI-7 against the ER-2 in-flight neutron measurements (Goldhagen 2004) and the NYC ground reference.
6. Separates the reference-dependent bias from the scale-invariant (structural) error.
7. Integrates fluence along seven great-circle routes (five from ICN, GRU-JNB and SYD-SCL) at FL370 to FL410 for 2009, 2019 and 2025 annual means, with monthly series.
8. Runs a sensitivity test with PARMA's aircraft-structure option.

## Requirements
- Python 3.13 with the packages in `requirements.txt`.
- PARMA 4.10 C++ source from JAEA EXPACS (https://phits.jaea.go.jp/expacs/). It is free for non-commercial use under JAEA's conditions, which require citing Sato 2015 and 2016 and sending JAEA a copy of the publication. The JAEA source is not redistributed here.
  1. Put the source in `parma/cpp/`.
  2. Copy it to `parma/build/` and add `code/seeflux_driver.cpp`. The WSL distro name defaults to `Ubuntu` and can be changed with `SEEFLUX_WSL_DISTRO`.
  3. Build in WSL Ubuntu 24.04 (g++ 13.3):
     `g++ -O2 -o seeflux_driver seeflux_driver.cpp subroutines.cpp`
- CARI-7, the FAA Windows executable (release of 15 November 2018, as stated in its HELP.TXT). It is used only by `bench_goldhagen.py`. Put it in `cari7/CARI_7_DVD/` and set `MENUS = NO!` in `CARI.INI`.
- Source PDFs for digitization (`data/refs/`) come from the publishers and are not redistributed. The digitized curves are provided in `data/step4/normand1996_fig*_digitized.csv`.

## Run
```
python code/analysis_main.py            # full run; the PARMA and CARI-7 steps added in revision (routes, device folds, cross-checks) take about 2 to 3 hours in total
python code/analysis_main.py --summary-only
```
Outputs:
- `data/experiment_summary.json`: all headline numbers
- `data/raw_results.csv`: the PARMA reference grid
- `data/step4/*.csv`: per-step tables
- `data/run_log.txt`: full log

Seeds are fixed per step: 20261014 (audit.py location sample, fit_formula.py bootstrap), 20261015 (validate_direct.py independent set), 20261016 (audit_revision.py bootstrap intervals, audit_revision2.py cross-validation folds, ml_analysis.py). The PARMA solar table used is input/FFPtable.day of the EXPACS C++ package (file date 2026-03-27, last entry 2026-03-26); the daily W at every independent point is also stored in data/step4/validation_direct.csv.

## Units and conventions
- Flux is in n cm⁻² h⁻¹, for E > 10 MeV unless the column says otherwise.
- W is PARMA's W-index (solar activity). Rc is the vertical cutoff rigidity in GV, from PARMA's `getrcpp`.
- Depth comes from altitude through the US Standard Atmosphere 1976.

## Third-party data
- Coastlines in Fig. 5: Natural Earth 1:110m coastline (public domain), https://github.com/nvkelso/natural-earth-vector, file `data/geo/ne_110m_coastline.geojson` (downloaded 2026-10-15).

## Additions after internal review (2026-10-16)
- `audit_revision.py`: shares at the reference-scale bounds k = 0.88, 1, 1.37; flat 5600/6000; hemisphere and latitude-band breakdown; bootstrap intervals; GLE-day sensitivity; date-matched JESD89 (two illustrative mappings); surrogate multipliers and their coverage.
- `audit_revision2.py`: attribution on the date-sampled cruise points; flat 6000 against PARMA neutrons + protons (driver keyword `PR`); bounding environment.
- `routes.py` (annual-mean W of 2019 and 2025, monthly series, storm-period day), `routes_levels.py` (seven routes incl. GRU-JNB and SYD-SCL, FL370/390/410, 2009/2019/2025), `ser_routes.py` (cross-section sensitivity, driver keyword `WC`), `cari_crosscheck.py` (CARI-7 check of the spread and the solar swing; needs the FAA CARI-7 distribution in `cari7/CARI_7_DVD`).
- `check_bsyd_tableA1.py`: printed BSYD equations against Gordon et al. Table A-I. `verify_parma.py`: build check against the JAEA samples and integration convergence.
- `solar_w.py`: daily, monthly and annual W from the PARMA solar table, GLE days flagged.

## Figures that do not need PARMA
All figure scripts in `figure/` read only the CSV/JSON outputs in `data/` (and `data/geo/` for coastlines), so Figs. 1 to 5 and S1 can be regenerated from the released outputs without compiling PARMA. Fig. 1 also writes `data/step4/fig1_stats.json`.
