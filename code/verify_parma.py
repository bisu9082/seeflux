# Reference-model checks quoted in Section II-B, stored in data/step4/parma_verification.json:
#  (1) the PARMA 4.10 build reproduces the sample spectra distributed by JAEA (SpecOut/*.out);
#  (2) the driver's >10 MeV integral is converged: 4000 vs 16000 log-spaced integration points.
import json, pathlib, subprocess, re
import numpy as np
from wslpath_util import wsl, DISTRO, run_driver

ROOT = pathlib.Path(__file__).resolve().parents[1]
CPP = wsl(ROOT / "parma" / "cpp")
TMP = "/tmp/seeflux_parma_verify"


def sh(cmd):
    return subprocess.run(["wsl", "-d", DISTRO, "--", "bash", "-lc", cmd], capture_output=True, text=True, check=True).stdout


def numbers(text):
    rows = []
    for line in text.splitlines():
        v = line.split()
        try:
            rows.append([float(x) for x in v])
        except ValueError:
            continue  # header or label line
    return rows


out = {"samples": {}}
for cond in ["Tokyo-Smin", "ERII-exp"]:
    sh(f'rm -rf {TMP} && cp -r "{CPP}" {TMP} && cd {TMP} && '
       f'sed -i \'s/string condition = "[^"]*";/string condition = "{cond}";/\' main.cpp && '
       f'g++ -O2 -o parma_main main.cpp subroutines.cpp && mv SpecOut/{cond}.out ref.out && ./parma_main > /dev/null')
    ref = numbers(sh(f"cat {TMP}/ref.out"))
    new = numbers(sh(f"cat {TMP}/SpecOut/{cond}.out"))
    assert len(ref) == len(new) and len(ref) > 0, (cond, len(ref), len(new))
    a, b = np.array(ref, float), np.array(new, float)
    rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
    # column 0 = energy, column 1 = neutron flux, then proton, ions, muons, e+/-, photon
    out["samples"][cond] = dict(rows=int(a.shape[0]), values=int(a.size),
                                neutron_max_rel_diff=float(rel[:, 1].max()),
                                other_columns_differing=sorted(set(np.where(rel > 1e-6)[1].tolist()) - {1}),
                                other_values_differing=int((rel[:, 2:] > 1e-6).sum()),
                                largest_differing_value=float(a[rel > 1e-6].max()) if (rel > 1e-6).any() else 0.0)

# (2) integration convergence of the >10 MeV integral (mode S: W, Rc, depth, g)
conds = [(0, 0, 200, 1), (150, 0, 200, 1), (75, 10, 250, 1), (0, 18, 120, 1), (150, 18, 450, 1), (75, 4, 1033, 1)]
lines = [f"c{i} S {w} {rc} {d} {g}" for i, (w, rc, d, g) in enumerate(conds)]
r4 = run_driver(lines)
build = wsl(ROOT / "parma" / "build")
r16 = subprocess.run(["wsl", "-d", DISTRO, "--", "bash", "-lc", f'cd "{build}" && SEEFLUX_NINT=16000 ./seeflux_driver'],
                     input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
r16 = [l.split() for l in r16.splitlines() if l.strip()]
v4 = np.array([[float(x) for x in row[1:]] for row in r4])
v16 = np.array([[float(x) for x in row[1:]] for row in r16])
out["integration"] = dict(n_conditions=len(conds), nint=[4000, 16000],
                          max_rel_diff=float(np.max(np.abs(v4 - v16) / np.maximum(np.abs(v16), 1e-300))))
json.dump(out, open(ROOT / "data/step4/parma_verification.json", "w"), indent=2)
print(json.dumps(out, indent=2))
