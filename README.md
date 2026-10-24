# seeflux

Code and data for

> S. Kim, M. Shin, J. Park and K. Kang, "Auditing the Boeing-Model Neutron Flux Procedure Referenced by IEC 62396-1 at Cruise Altitude: Cutoff Rigidity, Solar Phase and Device Response" (submitted).

The analysis compares the simplified atmospheric neutron flux estimates used in avionics single-event-effect (SEE) analyses with the PARMA model, over 2009 to 2026, on 2000 random flight conditions and seven routes. The estimates are the Boeing-model curves referenced by IEC 62396-1, the JESD89 ground scaling and vendor closed-form factors. The key results are checked with CARI-7.

## Contents

| Path | What it holds |
|---|---|
| `code/` | Every analysis step. `analysis_main.py` runs the full pipeline (30 steps) in order and writes `data/experiment_summary.json`. |
| `code/seeflux_driver.cpp` | Small C++ driver that calls PARMA 4.10 for integral fluxes, Weibull folds, band fluxes and protons. The PARMA source is **not** included (see below). |
| `code/README_code.md` | Step-by-step description of the scripts and their outputs. |
| `data/step4/` | All outputs that the tables, figures and numbers of the paper are read from: PARMA reference grid, evaluation sets, audits, routes, device folds, robustness checks. |
| `data/step_ml/` | Exploratory machine-learning analyses (plan, results, run logs). |
| `data/refs/` | Values transcribed from published tables (Cecchetto et al. 2019, Table 2 and 3; Gordon et al. 2004, Table A-I). |
| `data/geo/` | Natural Earth coastlines (public domain) for the route map. |
| `figure/` | Scripts that draw the figures from `data/step4`. |
| `data/a1_cleanroom_compare.csv` | Reproducibility check: a full rerun in a fresh copy reproduced all 62 outputs of `data/step4` bit for bit. |

## Requirements

* Python 3.13 with the packages in `code/requirements.txt`.
* **PARMA 4.10** (C++ version), distributed by the Japan Atomic Energy Agency with EXPACS: https://phits.jaea.go.jp/expacs/ . Obtain it under its own terms, place the source in `parma/build/` and compile it there together with `code/seeflux_driver.cpp`:
  `g++ -O2 -o seeflux_driver seeflux_driver.cpp subroutines.cpp`
  The scripts call the driver through WSL (Ubuntu); set `SEEFLUX_WSL_DISTRO` if your distribution has another name.
* **CARI-7** (FAA), Windows executable, version of 15 November 2018, placed in `cari7/CARI_7_DVD/`, for the CARI-7 steps (`bench_goldhagen.py`, `cari_crosscheck.py`, `cutoff_epoch.py`).
* Two published PDFs are needed only to redo the digitization (`digitize_normand.py`: Normand 1996, IEEE TNS 43, 461; `er2_gt10_anchor.py`: Goldhagen et al. 2004, Radiat. Prot. Dosim. 110, 387). They are not redistributed; the digitized results are in `data/step4`, so all later steps run without them.

## Running

```
python code/analysis_main.py
```

The full run takes about 3 to 4 hours, most of it in the route, device and CARI-7 steps. Single steps can be run on their own once their inputs exist in `data/step4`.

## Not included

PARMA and CARI-7 themselves, publisher PDFs and figure crops, the text of standards (IEC 62396-1, JESD89B, EASA CM-AS-004), and the manuscript. The released files contain PARMA outputs, not the PARMA source.

## Licence

Code: MIT licence (`LICENSE`). Data in `data/`: CC BY 4.0. Transcribed values in `data/refs/` remain attributable to the cited publications.

## Contact

Ku Kang (corresponding author), CBRN Defense Research Institute, Seoul, Republic of Korea.
