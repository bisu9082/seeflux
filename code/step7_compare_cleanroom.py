# Step 7 A1: compare clean-room outputs with project outputs, file by file.
# Usage: python step7_compare_cleanroom.py <cleanroom_root>
import sys, json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CR = pathlib.Path(sys.argv[1])
rows = []
for sub in ["data/step4", "data/step_ml"]:
    for f in sorted((ROOT / sub).glob("*")):
        g = CR / sub / f.name
        if f.suffix not in (".csv", ".json", ".npy"):
            continue
        if not g.exists():
            rows.append((sub + "/" + f.name, "MISSING in clean room", "")); continue
        if f.suffix == ".csv":
            a, b = pd.read_csv(f), pd.read_csv(g)
            if a.shape != b.shape or list(a.columns) != list(b.columns):
                rows.append((sub + "/" + f.name, "SHAPE/COLUMNS differ", f"{a.shape} vs {b.shape}")); continue
            num = a.select_dtypes("number").columns
            d = np.nanmax(np.abs(a[num].values - b[num].values) / np.maximum(np.abs(a[num].values), 1e-12)) if len(num) else 0.0
            txt = (a.drop(columns=num).astype(str).values != b.drop(columns=num).astype(str).values).sum()
            rows.append((sub + "/" + f.name, "identical" if d == 0 and txt == 0 else "DIFF", f"max rel {d:.2e}, text cells {txt}"))
        elif f.suffix == ".npy":
            a, b = np.load(f), np.load(g)
            d = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")
            rows.append((sub + "/" + f.name, "identical" if d == 0 else "DIFF", f"max abs {d:.2e}"))
        else:
            same = json.load(open(f)) == json.load(open(g))
            rows.append((sub + "/" + f.name, "identical" if same else "DIFF", ""))
df = pd.DataFrame(rows, columns=["file", "status", "detail"])
df.to_csv(ROOT / "data/step7/a1_cleanroom_compare.csv", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print("\nsummary:", df.status.value_counts().to_dict())
