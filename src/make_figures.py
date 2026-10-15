"""Figures for the evaluation report (matplotlib, static PNG)."""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, make_split

ROOT = os.environ.get("PROJ", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = os.path.join(ROOT, "results"); F = os.path.join(ROOT, "report", "figures"); os.makedirs(F, exist_ok=True)
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b9bdc2", "#1f2328", "#5f6670"
FS_LABEL = {"image_only": "Image only", "image_hour": "Image + hour", "image_pm_hour": "Image + PM2.5 + PM10 + hour"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c9ccd1", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#eceef0", "grid.linewidth": 0.8, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "figure.dpi": 150, "legend.frameon": False})

log = json.load(open(f"{R}/results.json")); preds = pd.read_csv(f"{R}/test_predictions.csv")
cvr = json.load(open(f"{R}/cv.json")) if os.path.exists(f"{R}/cv.json") else None
df = load(os.environ.get("DATA_CSV", "data/IND_and_Nep_AQI_Dataset.csv")); df["split"] = make_split(df)

# 1. dataset composition
g = df.groupby("Location").agg(images=("Filename", "size"), obs=("obs_id", "nunique")).sort_values("images")
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
ax[0].barh(g.index, g.images, color=BLUE, height=0.6); ax[0].set_title("Images per location"); ax[0].grid(axis="y", visible=False)
ax[1].barh(g.index, g.obs, color=ORANGE, height=0.6); ax[1].set_title("Distinct pollution observations per location"); ax[1].set_yticklabels([]); ax[1].grid(axis="y", visible=False)
for a, col in zip(ax, ["images", "obs"]):
    for y, v in enumerate(g[col]): a.text(v, y, f" {v}", va="center", fontsize=8, color=INK)
fig.tight_layout(); fig.savefig(f"{F}/dataset_composition.png"); plt.close(fig)

# 2. group size distribution
gs = df.groupby("obs_id").size()
fig, ax = plt.subplots(figsize=(6, 2.6)); ax.hist(gs, bins=30, color=BLUE, edgecolor="white", linewidth=1)
ax.set_xlabel("Images per observation group"); ax.set_ylabel("Observation groups"); ax.set_title("Many images share one pollution reading")
fig.tight_layout(); fig.savefig(f"{F}/group_sizes.png"); plt.close(fig)

# 3. actual vs predicted for the selected models
for target in ["NO2", "O3"]:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharex=True, sharey=True)
    sel = log["selected"][target]
    lim = 0
    for ax, fs in zip(axes, FS_LABEL):
        mname = sel[fs]["model"]; p = preds[(preds.target == target) & (preds.feature_set == fs) & (preds.model == mname)]
        gp = p.groupby("obs_id").agg(y=("y_true", "first"), p=("y_pred", "mean"))
        lim = max(lim, p.y_true.max() * 1.25, np.percentile(p.y_pred, 99.5), gp.p.max())
        ax.scatter(p.y_true, p.y_pred.clip(upper=None), s=4, color=GREY, alpha=0.35, linewidths=0, label="Single image")
        ax.scatter(gp.y, gp.p, s=34, color=BLUE, edgecolors="white", linewidths=1.2, label="Observation group (mean)", zorder=3)
        c = next(c for c in log["candidates"] if c["target"] == target and c["fset"] == fs and c["model"] == mname)
        ax.set_title(f"{FS_LABEL[fs]}\n{mname}", fontsize=9)
        ax.text(0.03, 0.97, f"group R² {c['test']['group']['R2']:.2f}\nimage R² {c['test']['image']['R2']:.2f}", transform=ax.transAxes, va="top", fontsize=8, color=INK)
        ax.set_xlabel(f"Measured {target}")
    nclip = 0
    for ax, fs in zip(axes, FS_LABEL):
        mname = sel[fs]["model"]; p = preds[(preds.target == target) & (preds.feature_set == fs) & (preds.model == mname)]
        nclip += int((p.y_pred > lim * 1.05).sum())
    for ax in axes:
        for coll in ax.collections[:1]:
            off = coll.get_offsets(); off[:, 1] = np.minimum(off[:, 1], lim * 1.04); coll.set_offsets(off)
        ax.plot([0, lim * 1.05], [0, lim * 1.05], color=MUTED, lw=1, ls="--", zorder=1); ax.set_xlim(0, lim * 1.05); ax.set_ylim(0, lim * 1.05)
    axes[0].set_ylabel(f"Predicted {target}"); axes[0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle(f"{target}: actual vs predicted on the held-out test set", fontweight="bold", fontsize=10.5)
    if nclip: fig.text(0.5, 0.005, f"{nclip} single-image predictions above the plot range are drawn at the top edge.", ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(); fig.savefig(f"{F}/scatter_{target}.png"); plt.close(fig)

# 4. MLP training / validation curves
H = log["histories"]
fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0))
for i, target in enumerate(["NO2", "O3"]):
    for j, fs in enumerate(FS_LABEL):
        ax = axes[i, j]; key = next(k for k in H if k.startswith(f"{target}|{fs}|"))
        h = pd.DataFrame(H[key]); be = h.val_loss.idxmin()
        ax.plot(h.epoch, h.train_loss, color=BLUE, lw=2, label="Training loss (image level)")
        ax.plot(h.epoch, h.val_loss, color=ORANGE, lw=2, label="Validation loss (group level)")
        ax.axvline(h.epoch[be], color=MUTED, lw=1, ls=":"); ax.text(h.epoch[be], ax.get_ylim()[1] * 0.95, f" best {h.epoch[be]}", fontsize=7.5, color=MUTED, va="top")
        ax.set_title(f"{target} · {FS_LABEL[fs]}\n({key.split('|')[2]} + MLP)", fontsize=8.5); ax.set_xlabel("Epoch")
        if j == 0: ax.set_ylabel("MSE (normalised log target)")
hd, lb = axes[0, 0].get_legend_handles_labels(); fig.legend(hd, lb, loc="lower center", ncol=2, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f"{F}/mlp_curves.png"); plt.close(fig)

# 5. ridge tuning curves (validation) for the selected backbone of the image-only model
T = pd.DataFrame(log.get("tuning", []))
if len(T):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for ax, target in zip(axes, ["NO2", "O3"]):
        bb = log["selected"][target]["image_only"]["backbone"]
        t = T[(T.target == target) & (T.fset == "image_only") & (T.backbone == bb) & (T["head"] == "ridge")].copy()
        t["k"] = t.hp.map(lambda h: h["n_pca"]); t["a"] = t.hp.map(lambda h: h["alpha"])
        ramp = ["#c6dcf6", "#9cc2ee", "#6fa5e4", "#2a78d6", "#1d5fae", "#123f78"]
        for col, (k, tt) in zip(ramp, t.groupby("k")):
            ax.plot(tt.a, tt.val_zmse_group, marker="o", ms=4, lw=2, color=col, label=f"{k} PCs")
        ax.set_xscale("log"); ax.set_xlabel("Ridge alpha"); ax.set_title(f"{target} · image only · {bb}")
        ax.set_ylabel("Validation group MSE (norm. log)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{F}/ridge_tuning.png"); plt.close(fig)

# 6. comparison of test group-level R2 (all candidates)
C = pd.DataFrame([{"target": c["target"], "model": c["model"], "fset": c["fset"], "family": c["family"],
                   "R2": c["test"]["group"]["R2"]} for c in log["candidates"]])
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharex=True)
order_fs = ["none", "hour_only", "pm_hour_only", "location", "image_only", "image_hour", "image_pm_hour"]
for ax, target in zip(axes, ["NO2", "O3"]):
    c = C[C.target == target].copy(); c["o"] = c.fset.map(order_fs.index); c = c.sort_values(["o", "model"], ascending=[False, True])
    BL = {"none": "Baseline", "hour_only": "Baseline", "pm_hour_only": "Baseline", "location": "Reference"}
    lab = [(FS_LABEL[f] if f in FS_LABEL else BL[f]) + " | " + m for f, m in zip(c.fset, c.model)]
    col = [BLUE if f.startswith("image") else (GREY if f != "location" else AQUA) for f in c.fset]
    ax.barh(range(len(c)), c.R2.clip(lower=-0.5), color=col, height=0.62)
    for y, v in enumerate(c.R2): ax.text(max(v, 0) + 0.02, y, f"{v:.2f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(lab if target == "NO2" else lab, fontsize=6.8)
    ax.axvline(0, color=MUTED, lw=0.8); ax.axvline(0.75, color=ORANGE, lw=1, ls="--"); ax.set_title(f"{target} · test R² (group level)")
    ax.grid(axis="y", visible=False); ax.set_xlim(-0.5, 1)
fig.tight_layout(); fig.savefig(f"{F}/test_r2_comparison.png"); plt.close(fig)

# 7. cross-validation summary
if cvr:
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    keys = ["pm_hour_only", "image_only", "image_hour", "image_pm_hour"]
    lab = {"pm_hour_only": "PM2.5+PM10+hour\n(no image)", **{k: FS_LABEL[k].replace(" + PM2.5 + PM10 + hour", " + PM + hour") for k in FS_LABEL}}
    x = np.arange(len(keys)); w = 0.36
    for i, (target, col) in enumerate([("NO2", BLUE), ("O3", ORANGE)]):
        m = [cvr[f"{target}|{k}"]["mean_sd"]["group"]["R2"][0] for k in keys]; s = [cvr[f"{target}|{k}"]["mean_sd"]["group"]["R2"][1] for k in keys]
        ax.bar(x + (i - 0.5) * w, m, w * 0.92, yerr=s, color=col, label=target, capsize=3, error_kw={"elinewidth": 1, "ecolor": INK})
        for xi, v in zip(x, m): ax.text(xi + (i - 0.5) * w, max(v, 0) + 0.04, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([lab[k] for k in keys], fontsize=8); ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Group-level R² (mean ± sd)"); ax.set_title("Repeated 5-fold grouped cross-validation (3 repeats)"); ax.legend()
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{F}/cv_r2.png"); plt.close(fig)

# 8. sample images
fig, axes = plt.subplots(2, 4, figsize=(9, 4.6))
locs = df.Location.unique()
import cv2
for ax, loc in zip(axes.flat, locs):
    f = df[df.Location == loc].Filename.iloc[len(df[df.Location == loc]) // 2]
    ax.imshow(cv2.cvtColor(cv2.imread(os.path.join(os.environ.get("IMG_DIR", "data/All_img"), f)), cv2.COLOR_BGR2RGB)); ax.set_title(loc, fontsize=8); ax.axis("off")
fig.tight_layout(); fig.savefig(f"{F}/sample_images.png"); plt.close(fig)
print("figures ->", F)
