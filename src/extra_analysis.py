"""Supplementary analyses: location-mean reference under the same CV folds, per-location test error,
and whether the selected image models track variation *within* a site (not just site identity)."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import df, fold_split, idx_of, evaluate, OUT, TARGETS
from models import metrics

res = {"cv_location_mean": {}, "within_location": {}, "per_location": {}}
for target in TARGETS:
    reps = []
    for r in range(3):
        folds = fold_split(1000 + r); preds = np.full(len(df), np.nan)
        for f in range(5):
            te = idx_of(folds == f, target); tr = idx_of(folds != f, target)
            g = df.iloc[tr].groupby(["Location", "obs_id"])[target].first()
            locm = g.groupby("Location").mean(); gm = g.mean()
            preds[te] = pd.Series(df.Location.values[te]).map(locm).fillna(gm).values
        ok = ~np.isnan(preds); reps.append(evaluate(np.where(ok)[0], preds[ok], target))
    res["cv_location_mean"][target] = {lvl: {m: [float(np.mean([x[lvl][m] for x in reps])), float(np.std([x[lvl][m] for x in reps]))]
                                             for m in ["MAE", "RMSE", "R2", "Pearson"]} for lvl in ["image", "group"]}

P = pd.read_csv(f"{OUT}/test_predictions.csv"); L = json.load(open(f"{OUT}/results.json"))
for target in TARGETS:
    for fs, sel in L["selected"][target].items():
        p = P[(P.target == target) & (P.feature_set == fs) & (P.model == sel["model"])]
        g = p.groupby("obs_id").agg(loc=("Location", "first"), y=("y_true", "first"), p=("y_pred", "mean"))
        # demeaned by location (using test-group means): does the model rank observations within a site?
        g["yd"] = g.y - g.groupby("loc").y.transform("mean"); g["pd"] = g.p - g.groupby("loc").p.transform("mean")
        multi = g[g.groupby("loc").y.transform("size") >= 3]
        r = float(np.corrcoef(multi.yd, multi.pd)[0, 1]) if len(multi) > 3 else float("nan")
        res["within_location"][f"{target}|{fs}"] = {"model": sel["model"], "n_groups": int(len(multi)),
                                                     "n_locations": int(multi["loc"].nunique()), "within_site_pearson": r}
        res["per_location"][f"{target}|{fs}"] = {loc: {"n_groups": int(len(gg)), "MAE": float(np.abs(gg.p - gg.y).mean())} for loc, gg in g.groupby("loc")}
json.dump(res, open(f"{OUT}/extra_analysis.json", "w"), indent=1)
print(json.dumps(res["cv_location_mean"], indent=0)[:600]); print(json.dumps(res["within_location"], indent=1))
