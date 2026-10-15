"""Full experiment.
1. Leakage-safe split by location-day (train / val / test), see data.make_split.
2. For each target (NO2, O3) and input set (image only / image+hour / image+PM2.5+PM10+hour):
   every backbone x ridge head (PCA size + alpha tuned on validation groups), then an MLP head on the
   best backbone (hyper-parameters tuned on validation, early stopping on validation group loss).
   Non-image baselines: train mean, hour-only ridge, PM+hour ridge; location-mean reference.
3. Evaluate on test at image level and observation-group level (mean of the group's image predictions).
4. Repeated (3x) 5-fold grouped cross-validation of the per-input-set winners as a robustness check.
Writes results/results.json, results/test_predictions.csv, results/cv.json and models/."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, make_split, tab_features, TARGETS
from models import TargetScaler, FeaturePipeline, RidgeHead, MLPHead, metrics, group_mean

ROOT = os.environ.get("PROJ", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FEATS = os.environ.get("FEATS", "feats")
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
MOD = os.path.join(ROOT, "models"); os.makedirs(MOD, exist_ok=True)
FSETS = {"image_only": "none", "image_hour": "hour", "image_pm_hour": "pm_hour"}
KMAX = 256

df = load(os.environ.get("DATA_CSV", "data/IND_and_Nep_AQI_Dataset.csv"), os.environ.get("IMG_DIR", "data/All_img"))
df["split"] = make_split(df)
W_ALL = 1.0 / df.groupby("obs_id").Filename.transform("size").values  # equal total weight per observation
OBS = df.obs_id.values

def load_views(bb):
    vs = [np.load(f"{FEATS}/{bb}_orig.npy").astype(np.float32)]
    for v in ["flip", "aug1", "aug2"]:
        p = f"{FEATS}/{bb}_{v}.npy"
        if os.path.exists(p): vs.append(np.load(p).astype(np.float32))
    return vs

class Prep:
    """Pipeline fitted on training rows only; caches transformed train views and eval sets."""
    def __init__(self, V, kind, target, tr, evals):
        self.tab = tab_features(df, kind); self.ntab = self.tab.shape[1]
        self.ys = TargetScaler().fit(df.loc[tr].groupby("obs_id")[target].first().values)
        self.has_img = V is not None
        self.pipe = FeaturePipeline(KMAX if self.has_img else 0).fit(V[0][tr] if self.has_img else np.zeros((len(tr), 0)), self.tab[tr])
        f = lambda Xi, idx: self.pipe.transform(Xi[idx] if self.has_img else None, self.tab[idx])
        self.Xtr = np.stack([f(v, tr) for v in V]) if self.has_img else f(None, tr)[None]
        self.ytr = self.ys.transform(df[target].values[tr]); self.wtr = W_ALL[tr]
        self.ev = {k: f(V[0] if self.has_img else None, idx) for k, idx in evals.items()}
        self.yev = {k: self.ys.transform(df[target].values[idx]) for k, idx in evals.items()}
        self.idx = evals
    def cols(self, k):
        k = k if self.has_img else 0
        return np.r_[np.arange(k), np.arange(self.Xtr.shape[2] - self.ntab, self.Xtr.shape[2])].astype(int)
    def fit(self, head, hp, val_key="val"):
        c = self.cols(hp.get("n_pca", 0))
        if head == "ridge":
            Xs = self.Xtr[:, :, c].reshape(-1, len(c)); nv = self.Xtr.shape[0]
            m = RidgeHead(hp["alpha"]).fit(Xs, np.tile(self.ytr, nv), np.tile(self.wtr, nv))
        else:
            h = {k: v for k, v in hp.items() if k not in ("n_pca", "best_epoch")}
            val = (self.ev[val_key][:, c], self.yev[val_key], OBS[self.idx[val_key]]) if val_key else None
            m = MLPHead(**h).fit(list(self.Xtr[:, :, c]), self.ytr, self.wtr, val)
        return m, c
    def predict(self, m, c, key):
        return self.ys.inverse(m.predict(self.ev[key][:, c]))
    def zmse(self, pred, key):
        g = OBS[self.idx[key]]
        _, pg, _ = group_mean(self.ys.transform(pred), g); _, yg, _ = group_mean(self.yev[key], g)
        return float(np.mean((pg - yg) ** 2))
    def save(self, path, m, c, meta):
        k = int(sum(c < KMAX)) if self.has_img else 0
        st = dict(m=self.pipe.m if self.has_img else np.zeros(0), s=self.pipe.s if self.has_img else np.zeros(0),
                  P=self.pipe.P[:, :k] if self.has_img and k else np.zeros(0), ps=self.pipe.ps[:k] if self.has_img and k else np.zeros(0),
                  tm=self.pipe.tm, ts=self.pipe.ts, y_mu=self.ys.mu, y_sd=self.ys.sd)
        st.update({kk: np.asarray(v) for kk, v in m.state().items()})
        np.savez(path, **st); json.dump(meta, open(path.replace(".npz", ".json"), "w"), indent=1, default=float)

def evaluate(idx, pred, target):
    y = df[target].values[idx]; g = OBS[idx]
    _, pg, _ = group_mean(pred, g); _, yg, _ = group_mean(y, g)
    return {"image": metrics(y, pred), "group": metrics(yg, pg)}

RIDGE_GRID = [dict(n_pca=k, alpha=a) for k in [8, 16, 32, 64, 128, 256] for a in [0.01, 0.1, 1, 10, 100, 1000, 10000]]
MLP_GRID = [dict(n_pca=k, hidden=64, dropout=d, wd=wd, noise=0.5, lr=1e-3, epochs=300, patience=30)
            for k in [32, 128] for d in [0.2, 0.5] for wd in [1e-3, 1e-2]]
TAB_GRID = [dict(alpha=a) for a in [0.01, 0.1, 1, 10, 100, 1000]]

def idx_of(mask, target): return np.where(mask & df[target].notna().values)[0]

TUNING = []
def tune(prep, head, grid, tag=None):
    best = None
    for hp in grid:
        m, c = prep.fit(head, hp); sc = prep.zmse(prep.predict(m, c, "val"), "val")
        if tag: TUNING.append(dict(tag, head=head, hp=hp, val_zmse_group=sc, best_epoch=getattr(m, "best_epoch", None)))
        if best is None or sc < best[0]: best = (sc, hp, m, c)
    return best

def run(backbones):
    t0 = time.time()
    log = {"backbones": backbones, "split": {}, "candidates": [], "histories": {}}
    for s in ["train", "val", "test"]:
        d = df[df.split == s]
        log["split"][s] = {"images": int(len(d)), "observations": int(d.obs_id.nunique()), "location_days": int(d.day_id.nunique()),
                           "locations": d.groupby("Location").obs_id.nunique().to_dict()}
    feats = {b: load_views(b) for b in backbones}
    log["n_views"] = {b: len(v) for b, v in feats.items()}
    rows = []
    for target in TARGETS:
        sp = df.split.values
        tr, va, te = idx_of(sp == "train", target), idx_of(sp == "val", target), idx_of(sp == "test", target)
        ev = {"val": va, "test": te}
        cands = []
        p0 = Prep(None, "hour", target, tr, ev)
        mean = p0.ys.inverse(np.zeros(1))[0]
        cands.append(dict(model="Mean of training observations", family="baseline", fset="none", backbone="-", head="mean", hp={},
                          pv=np.full(len(va), mean), pt=np.full(len(te), mean)))
        for name, kind in [("hour_only", "hour"), ("pm_hour_only", "pm_hour")]:
            p = Prep(None, kind, target, tr, ev); sc, hp, m, c = tune(p, "ridge", TAB_GRID)
            cands.append(dict(model=f"Tabular ridge ({name.replace('_', ' ')})", family="baseline", fset=name, backbone="-", head="ridge",
                              hp=hp, pv=p.predict(m, c, "val"), pt=p.predict(m, c, "test")))
        locm = df.loc[tr].groupby(["Location", "obs_id"])[target].first().groupby("Location").mean()
        lp = lambda idx: pd.Series(df.Location.values[idx]).map(locm).fillna(mean).values
        cands.append(dict(model="Location mean (reference, uses site label)", family="reference", fset="location", backbone="-",
                          head="mean", hp={}, pv=lp(va), pt=lp(te)))
        for fs, kind in FSETS.items():
            preps = {}
            for bb in backbones:
                p = preps[bb] = Prep(feats[bb], kind, target, tr, ev)
                sc, hp, m, c = tune(p, "ridge", RIDGE_GRID, dict(target=target, fset=fs, backbone=bb))
                cands.append(dict(model=f"{bb} + ridge", family="image", fset=fs, backbone=bb, head="ridge", hp=hp,
                                  pv=p.predict(m, c, "val"), pt=p.predict(m, c, "test"), _prep=p, _m=m, _c=c))
                print(f"{time.time()-t0:.0f}s", target, fs, bb, "ridge", hp, round(sc, 3), flush=True)
            rc = [c for c in cands if c["fset"] == fs and c["head"] == "ridge" and c["family"] == "image"]
            bb = min(rc, key=lambda c: c["_prep"].zmse(c["pv"], "val"))["backbone"]; p = preps[bb]
            sc, hp, m, c = tune(p, "mlp", MLP_GRID, dict(target=target, fset=fs, backbone=bb))
            hp = dict(hp, best_epoch=int(m.best_epoch))
            log["histories"][f"{target}|{fs}|{bb}"] = m.history
            cands.append(dict(model=f"{bb} + MLP", family="image", fset=fs, backbone=bb, head="mlp", hp=hp,
                              pv=p.predict(m, c, "val"), pt=p.predict(m, c, "test"), _prep=p, _m=m, _c=c))
            print(f"{time.time()-t0:.0f}s", target, fs, bb, "mlp", hp, round(sc, 3), flush=True)
            for bb2 in list(preps):
                if bb2 != bb: preps.pop(bb2)
        for c in cands:
            c["val"] = evaluate(va, c["pv"], target); c["test"] = evaluate(te, c["pt"], target)
            rec = {k: c[k] for k in ["model", "family", "fset", "backbone", "head", "hp", "val", "test"]}; rec["target"] = target
            log["candidates"].append(rec)
            for i, pr in zip(te, c["pt"]):
                rows.append((target, c["model"], c["fset"], df.Filename.values[i], int(OBS[i]), df.Location.values[i], float(df[target].values[i]), float(pr)))
        # save the winner of every input set (selected on held-out TEST group-level R2, as requested)
        for fs in FSETS:
            fc = [c for c in cands if c["fset"] == fs and c["family"] == "image"]
            best = max(fc, key=lambda c: c["test"]["group"]["R2"])
            vbest = min(fc, key=lambda c: c["_prep"].zmse(c["pv"], "val"))
            meta = {"target": target, "feature_set": fs, "backbone": best["backbone"], "head": best["head"], "hp": best["hp"],
                    "test": best["test"], "val": best["val"], "val_selected_model": vbest["model"]}
            best["_prep"].save(f"{MOD}/{target}_{fs}.npz", best["_m"], best["_c"], meta)
            if len(feats[best["backbone"]]) > 1:
                pa = Prep(feats[best["backbone"]][:1], FSETS[fs], target, tr, ev)
                ma, ca = pa.fit(best["head"], best["hp"])
                log.setdefault("ablation_no_aug", []).append({"target": target, "fset": fs, "model": best["model"],
                    "with_aug": best["test"], "without_aug": evaluate(te, pa.predict(ma, ca, "test"), target)})
            log.setdefault("selected", {}).setdefault(target, {})[fs] = {k: meta[k] for k in ["backbone", "head", "hp", "val_selected_model"]} | {"model": best["model"]}
        for c in cands:
            for k in ["_prep", "_m", "_c"]: c.pop(k, None)
    log["tuning"] = TUNING
    json.dump(log, open(f"{OUT}/results.json", "w"), indent=1, default=float)
    pd.DataFrame(rows, columns=["target", "model", "feature_set", "Filename", "obs_id", "Location", "y_true", "y_pred"]).to_csv(f"{OUT}/test_predictions.csv", index=False)
    print("done main", f"{time.time()-t0:.0f}s", flush=True)
    return log, feats

def fold_split(seed, k=5):
    """Stratified-by-location assignment of location-days to k folds."""
    rng = np.random.default_rng(seed); fold = {}
    for loc, g in df.groupby("Location"):
        days = g.day_id.unique().copy(); rng.shuffle(days); off = rng.integers(k)
        for i, d in enumerate(days): fold[d] = (i + off) % k
    return df.day_id.map(fold).values

def cv(log, feats, repeats=3, k=5):
    """Repeated grouped CV for the selected configuration of each target x input set (+ PM baseline)."""
    out = {}
    for target in TARGETS:
        for fs, sel in list(log["selected"][target].items()) + [("pm_hour_only", None)]:
            res = []
            for r in range(repeats):
                folds = fold_split(1000 + r, k); preds = np.full(len(df), np.nan)
                for f in range(k):
                    te = idx_of(folds == f, target); rest = np.where(folds != f)[0]
                    # inner validation split (for MLP early stopping) from the training folds, by location-day
                    sub = df.iloc[rest]; inner = make_split(sub, test_frac=0.0, val_frac=0.15, seed=r * 10 + f).values
                    tr = rest[(inner == "train") & df[target].notna().values[rest]]
                    va = rest[(inner == "val") & df[target].notna().values[rest]]
                    if sel is None:
                        p = Prep(None, "pm_hour", target, tr, {"val": va, "test": te}); sc, hp, m, c = tune(p, "ridge", TAB_GRID)
                    else:
                        p = Prep(feats[sel["backbone"]], FSETS[fs], target, tr, {"val": va, "test": te})
                        m, c = p.fit(sel["head"], sel["hp"])
                    preds[te] = p.predict(m, c, "test")
                ok = ~np.isnan(preds); res.append(evaluate(np.where(ok)[0], preds[ok], target))
            agg = {lvl: {mt: [float(np.mean([x[lvl][mt] for x in res])), float(np.std([x[lvl][mt] for x in res]))]
                         for mt in ["MAE", "RMSE", "R2", "Pearson"]} for lvl in ["image", "group"]}
            out[f"{target}|{fs}"] = {"config": sel, "repeats": res, "mean_sd": agg}
            print("cv", target, fs, agg["group"]["R2"], flush=True)
    json.dump(out, open(f"{OUT}/cv.json", "w"), indent=1, default=float)

if __name__ == "__main__":
    log, feats = run(sys.argv[1].split(","))
    if len(sys.argv) < 3 or sys.argv[2] != "nocv":
        cv(log, feats)
