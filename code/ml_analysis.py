# Exploratory ML analysis (plan fixed in data/step_ml/ml_plan.md before running). Inputs: Step 4 outputs only.
#   ML1 emulator benchmark against Eq. (1)      ML2 error attribution of the simplified models
#   ML3 rule for IEC non-conservatism           ML4 sparse symbolic regression
import json, pathlib, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import balanced_accuracy_score, recall_score, r2_score
import xgboost as xgb, shap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from seeflux_formula import flux_gt10, load_params

warnings.filterwarnings("ignore")
SEED = 20261016
ROOT = pathlib.Path(__file__).resolve().parents[1]
S4, OUT = ROOT / "data/step4", ROOT / "data/step_ml"
OUT.mkdir(exist_ok=True)
X3 = ["alt_km", "Rc", "W"]

grid = pd.read_csv(S4 / "parma_grid.csv")
grid = grid[grid.alt_km >= 6.0].reset_index(drop=True)
TR_ALT = [6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0]
TR_RC = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
TR_W = [0, 75, 150]
is_tr = grid.alt_km.isin(TR_ALT) & grid.Rc.isin(TR_RC) & grid.W.isin(TR_W)
tr, te = grid[is_tr].reset_index(drop=True), grid[~is_tr].reset_index(drop=True)
val = pd.read_csv(S4 / "validation_direct.csv")
ind = val[val.in_domain].reset_index(drop=True)
assert len(tr) == 252 and len(ind) == 1995, (len(tr), len(ind))


def errstats(pred_ln, F):
    e = np.abs(np.exp(pred_ln) / F - 1) * 100
    return dict(median_pct=round(float(np.median(e)), 2), p95_pct=round(float(np.percentile(e, 95)), 2),
                max_pct=round(float(e.max()), 2))


# ---------------------------------------------------------------- ML1 emulators
P_EQ1 = load_params()


class Eq1:  # fixed closed form; refitted only in region hold-outs
    name = "Eq. (1)"
    def fit(self, X, y):
        from scipy.optimize import least_squares
        f = lambda p: np.log(flux_gt10(X[:, 0], X[:, 1], X[:, 2], p)) - y
        self.p = least_squares(f, P_EQ1, method="lm").x
        return self
    def predict(self, X):
        return np.log(flux_gt10(X[:, 0], X[:, 1], X[:, 2], self.p))


class MLPEns:
    def __init__(self, n=5):
        self.n = n
    def fit(self, X, y):
        self.sx, self.ym, self.ys = StandardScaler().fit(X), y.mean(), y.std()
        self.ms = [MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", solver="lbfgs", alpha=1e-4, max_iter=20000,
                                random_state=SEED + i).fit(self.sx.transform(X), (y - self.ym) / self.ys)
                   for i in range(self.n)]
        return self
    def predict(self, X):
        Z = self.sx.transform(X)
        return np.mean([m.predict(Z) for m in self.ms], axis=0) * self.ys + self.ym


class GP:
    def fit(self, X, y):
        self.sx, self.ym = StandardScaler().fit(X), y.mean()
        k = ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0, 1.0], nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-2))
        self.m = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=3, random_state=SEED)
        self.m.fit(self.sx.transform(X), y - self.ym)
        return self
    def predict(self, X, return_std=False):
        mu, sd = self.m.predict(self.sx.transform(X), return_std=True)
        return (mu + self.ym, sd) if return_std else mu + self.ym


def make_models():
    return {
        "Eq. (1)": Eq1(),
        "Ridge, cubic poly": make_pipeline(StandardScaler(), PolynomialFeatures(3), Ridge(1e-6)),
        "Gaussian process": GP(),
        "XGBoost": xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=4, subsample=0.8,
                                    random_state=SEED, verbosity=0),
        "MLP ensemble": MLPEns(),
        "kNN (k=4, distance)": make_pipeline(StandardScaler(), KNeighborsRegressor(4, weights="distance")),
    }


def ml1():
    rows, cover = [], []
    ytr = np.log(tr.F_gt10.values)
    for name, m in make_models().items():
        t0 = time.time()
        m.fit(tr[X3].values, ytr)
        r = dict(model=name, test="held-out grid (4669)", **errstats(m.predict(te[X3].values), te.F_gt10.values))
        rows.append(r)
        rows.append(dict(model=name, test="independent (1995)", **errstats(m.predict(ind[X3].values), ind.PARMA.values)))
        if name == "Gaussian process":
            mu, sd = m.predict(ind[X3].values, return_std=True)
            z = np.abs(mu - np.log(ind.PARMA.values)) / sd
            cover.append(dict(test="independent", coverage95=round(float(np.mean(z < 1.96)), 4),
                              median_sd_pct=round(float(np.median(sd) * 100), 2)))
        print(f"  ML1 {name:22s} {time.time() - t0:5.1f}s")
    regions = [("W>=100 (train W<=75)", tr.W < 150, grid.W >= 100),
               ("Rc>12 (train Rc<=12)", tr.Rc <= 12, grid.Rc > 12),
               ("alt>12.5 (train alt<=12)", tr.alt_km <= 12, grid.alt_km > 12.5),
               ("alt<9 (train alt>=9)", tr.alt_km >= 9, grid.alt_km < 9)]
    for rname, keep, test in regions:
        G = grid[test]
        for name, m in make_models().items():
            m.fit(tr[keep][X3].values, np.log(tr[keep].F_gt10.values))
            rows.append(dict(model=name, test="extrapolate " + rname, **errstats(m.predict(G[X3].values), G.F_gt10.values)))
            if name == "Gaussian process":
                mu, sd = m.predict(G[X3].values, return_std=True)
                z = np.abs(mu - np.log(G.F_gt10.values)) / sd
                cover.append(dict(test="extrapolate " + rname, coverage95=round(float(np.mean(z < 1.96)), 4),
                                  median_sd_pct=round(float(np.median(sd) * 100), 2)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ml1_emulators.csv", index=False)
    pd.DataFrame(cover).to_csv(OUT / "ml1_gp_coverage.csv", index=False)
    return df, pd.DataFrame(cover)


# ---------------------------------------------------------------- ML2 attribution
MODELS = {"S1_IEC": "IEC 62396-1", "S2_JESD_mid": "JESD89", "S4_AN_ABB": "AN4589 ABB", "S4b_AN_CoolCAD": "AN4589 CoolCAD"}


def ml2():
    A = pd.read_csv(S4 / "audit_nodes.csv")
    A = A[A.alt_km.between(9.0, 12.5)].reset_index(drop=True)
    A["abslat"] = A.lat.abs()
    rng = np.random.default_rng(SEED)
    # split by location (all altitudes and W levels of a grid cell stay on one side), so that neighbouring nodes of
    # the deterministic grid do not leak between fit and test (Step 8 review, R2)
    loc = A.lat.astype(str) + "_" + A.lon.astype(str)
    keep = set(rng.permutation(loc.unique())[: loc.nunique() // 2])
    half = loc.isin(keep).values
    sets = [("alt,|lat|", ["alt_km", "abslat"]), ("alt,|lat|,W", ["alt_km", "abslat", "W"]),
            ("alt,|lat|,Rc", ["alt_km", "abslat", "Rc"]), ("alt,|lat|,W,Rc", ["alt_km", "abslat", "W", "Rc"])]
    rows, shp = [], []
    for col, lab in MODELS.items():
        y = np.log(A[col] / A.PARMA).values
        r = dict(model=lab, n=len(A), var_lny=round(float(y.var()), 4))
        for sname, feats in sets:
            m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, random_state=SEED, verbosity=0)
            m.fit(A.loc[half, feats], y[half])
            r["R2 " + sname] = round(float(r2_score(y[~half], m.predict(A.loc[~half, feats]))), 6)
        rows.append(r)
        feats = sets[-1][1]
        m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, random_state=SEED, verbosity=0)
        m.fit(A.loc[half, feats], y[half])
        sub = A.loc[~half, feats].sample(5000, random_state=SEED)
        sv = np.abs(shap.TreeExplainer(m).shap_values(sub)).mean(axis=0)
        shp.append(dict(model=lab, **{f: round(float(v / sv.sum()), 6) for f, v in zip(feats, sv)}))
    a, s = pd.DataFrame(rows), pd.DataFrame(shp)
    a.to_csv(OUT / "ml2_attribution_r2.csv", index=False)
    s.to_csv(OUT / "ml2_shap_shares.csv", index=False)
    return a, s


# ---------------------------------------------------------------- ML3 non-conservative rule
def ml3():
    A = pd.read_csv(S4 / "audit_nodes.csv")
    A = A[A.alt_km.between(9.0, 12.5)].reset_index(drop=True)
    V = val[val.alt_km.between(9.0, 12.5)].reset_index(drop=True)
    for D in (A, V):
        D["abslat"] = D.lat.abs()
    ya = (A.S1_IEC / A.PARMA < 1).astype(int).values
    yv = (V.S1_IEC / V.PARMA < 1).astype(int).values
    rows, rules = [], {}
    for fname, feats in [("A: alt,|lat|,W", ["alt_km", "abslat", "W"]), ("B: alt,Rc,W", ["alt_km", "Rc", "W"])]:
        cands = {f"tree depth {d}": DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=SEED)
                 for d in (2, 3)}
        cands["logistic"] = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        for mname, m in cands.items():
            m.fit(A[feats], ya)
            p = m.predict(V[feats])
            rows.append(dict(features=fname, model=mname, n_test=len(V), base_rate=round(float(yv.mean()), 3),
                             balanced_acc=round(float(balanced_accuracy_score(yv, p)), 4),
                             recall_noncons=round(float(recall_score(yv, p)), 4),
                             flagged_share=round(float(p.mean()), 4)))
            if mname.startswith("tree"):
                rules[f"{fname} | {mname}"] = export_text(m, feature_names=feats, decimals=1)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ml3_noncons_rule.csv", index=False)
    (OUT / "ml3_tree_rules.txt").write_text("\n\n".join(f"== {k}\n{v}" for k, v in rules.items()), encoding="utf-8")
    return df, rules


# ---------------------------------------------------------------- ML4 sparse symbolic regression
def library(D):
    h, rc, w = D.alt_km.values, D.Rc.values, D.W.values
    L = np.log1p(rc)
    return pd.DataFrame({
        "1": np.ones_like(h), "h": h, "h^2": h ** 2, "h^3": h ** 3,
        "ln(1+Rc)": L, "ln(1+Rc)^2": L ** 2, "ln(1+Rc)^3": L ** 3, "Rc": rc, "Rc^2": rc ** 2,
        "h ln(1+Rc)": h * L, "h^2 ln(1+Rc)": h ** 2 * L, "h ln(1+Rc)^2": h * L ** 2,
        "W": w, "W^2": w ** 2, "W h": w * h, "W ln(1+Rc)": w * L, "W Rc": w * rc,
        "W/(1+0.2Rc)": w / (1 + 0.2 * rc), "W h ln(1+Rc)": w * h * L, "ln(1+Rc^2)": np.log1p(rc ** 2),
    })


def ml4():
    Ltr, Lte, Lind = library(tr), library(te), library(ind)
    y = np.log(tr.F_gt10.values)
    chosen, path = ["1"], []
    while len(chosen) < 12:
        best = None
        for c in Ltr.columns:
            if c in chosen:
                continue
            cols = chosen + [c]
            b, *_ = np.linalg.lstsq(Ltr[cols].values, y, rcond=None)
            rss = float(np.sum((Ltr[cols].values @ b - y) ** 2))
            if best is None or rss < best[0]:
                best = (rss, c, b)
        rss, c, b = best
        chosen = chosen + [c]
        n, k = len(y), len(chosen)
        path.append(dict(k=k, added=c, BIC=round(n * np.log(rss / n) + k * np.log(n), 1),
                         **{"heldout_" + kk: v for kk, v in errstats(Lte[chosen].values @ b, te.F_gt10.values).items()},
                         **{"indep_" + kk: v for kk, v in errstats(Lind[chosen].values @ b, ind.PARMA.values).items()},
                         terms=" + ".join(chosen)))
    df = pd.DataFrame(path)
    eq1 = Eq1(); eq1.p = P_EQ1
    n = len(y); rss1 = float(np.sum((eq1.predict(tr[X3].values) - y) ** 2))
    ref = dict(k=8, added="Eq. (1), 8 params (nonlinear)", BIC=round(n * np.log(rss1 / n) + 8 * np.log(n), 1),
               **{"heldout_" + kk: v for kk, v in errstats(eq1.predict(te[X3].values), te.F_gt10.values).items()},
               **{"indep_" + kk: v for kk, v in errstats(eq1.predict(ind[X3].values), ind.PARMA.values).items()}, terms="Eq. (1)")
    df = pd.concat([df, pd.DataFrame([ref])], ignore_index=True)
    df.to_csv(OUT / "ml4_symbolic_path.csv", index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    t0 = time.time()
    d1, c1 = ml1(); print(d1.to_string(index=False)); print(c1.to_string(index=False))
    a2, s2 = ml2(); print(a2.to_string(index=False)); print(s2.to_string(index=False))
    d3, r3 = ml3(); print(d3.to_string(index=False)); [print("==", k, "\n" + v) for k, v in r3.items()]
    d4 = ml4(); print(d4.drop(columns="terms").to_string(index=False)); print(d4[["k", "terms"]].tail(4).to_string(index=False))
    json.dump(dict(seed=SEED, runtime_s=round(time.time() - t0, 1), plan="data/step_ml/ml_plan.md"),
              open(OUT / "ml_run_meta.json", "w"), indent=2)
    print("done in %.0fs" % (time.time() - t0))
