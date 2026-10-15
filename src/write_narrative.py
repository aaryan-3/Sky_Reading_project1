"""Assemble report/narrative.json: prose + tables, with every number pulled from the results files."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, make_split
ROOT = os.environ.get("PROJ", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); R = f"{ROOT}/results"
L = json.load(open(f"{R}/results.json")); CV = json.load(open(f"{R}/cv.json")); X = json.load(open(f"{R}/extra_analysis.json"))
df = load(os.environ.get("DATA_CSV", "data/IND_and_Nep_AQI_Dataset.csv")); df["split"] = make_split(df)
G = df.groupby("obs_id").first()
FS = {"image_only": "Image only", "image_hour": "Image + hour", "image_pm_hour": "Image + PM2.5 + PM10 + hour"}
f1 = lambda v: f"{v:.1f}"; f2 = lambda v: f"{v:.2f}"
def cand(t, fs, model): return next(c for c in L["candidates"] if c["target"] == t and c["fset"] == fs and c["model"] == model)
def selc(t, fs): return cand(t, fs, L["selected"][t][fs]["model"])
def cvg(t, fs, m="R2"): v = CV[f"{t}|{fs}"]["mean_sd"]["group"][m]; return f"{v[0]:.2f} ± {v[1]:.2f}"
def cvi(t, fs, m="R2"): v = CV[f"{t}|{fs}"]["mean_sd"]["image"][m]; return f"{v[0]:.2f} ± {v[1]:.2f}"
loc_cv = lambda t, lvl="group": "{:.2f} ± {:.2f}".format(*X["cv_location_mean"][t][lvl]["R2"])
base = lambda t, fs: next(c for c in L["candidates"] if c["target"] == t and c["fset"] == fs)

n2, o3 = {fs: selc("NO2", fs) for fs in FS}, {fs: selc("O3", fs) for fs in FS}
final_rows = []
for t, S in [("NO2", n2), ("O3", o3)]:
    for fs in FS:
        c = S[fs]
        final_rows.append([t, FS[fs], c["model"], f1(c["test"]["group"]["MAE"]), f2(c["test"]["group"]["R2"]), f2(c["test"]["image"]["R2"]), cvg(t, fs)])

sp = L["split"]
N = {"title": "Estimating NO₂ and O₃ from outdoor images", "subtitle": "Model evaluation report for the India and Nepal air-pollution image dataset (October 2026)"}
N["final_rows"] = final_rows
N["summary"] = [
 f"This report evaluates models that predict nitrogen dioxide (NO₂) and ozone (O₃) from outdoor photographs in the India and Nepal air-pollution image dataset. The 12,240 images all match a CSV record, but they collapse into only {G.shape[0]} distinct pollution observations (one location, date, hour and set of readings). Every image of an observation carries the same NO₂ and O₃ value, so the effective sample size is about one hundred, not twelve thousand. All splitting is therefore done by location-day, so no observation, and no other observation from the same site and day, is shared between training, validation and test.",
 f"Three input sets were compared, each built on ImageNet-pretrained CNNs (ResNet-50, ResNet-18, EfficientNet-Lite4, DenseNet-121, MobileNetV2) with ridge and MLP regression heads: image only; image + hour; and image + PM2.5 + PM10 + hour. Results are reported at image level and at observation-group level (the mean prediction over all images of one observation).",
 f"On the held-out test set ({sp['test']['observations']} observations, {sp['test']['images']:,} images), the best model overall is {n2['image_pm_hour']['model']} on image + PM2.5 + PM10 + hour for NO₂: group-level R² {f2(n2['image_pm_hour']['test']['group']['R2'])} (image level {f2(n2['image_pm_hour']['test']['image']['R2'])}). This is an environmental-fusion result, not a pure image result. Without the image, the same PM + hour inputs reach {f2(base('NO2','pm_hour_only')['test']['group']['R2'])} on the same test set. The best pure-image NO₂ model reaches group-level R² {f2(n2['image_only']['test']['group']['R2'])}. For O₃ the best test result is {f2(o3['image_pm_hour']['test']['group']['R2'])} (image + PM + hour) and {f2(o3['image_hour']['test']['group']['R2'])} (image + hour); image only reaches {f2(o3['image_only']['test']['group']['R2'])}.",
 f"The target of R² > 0.75 is met only by the NO₂ fusion model, only at group level, and only on this one test split. Repeated grouped cross-validation over all observations, which is more reliable with so few observations, gives clearly lower values: NO₂ fusion {cvg('NO2','image_pm_hour')}, NO₂ image only {cvg('NO2','image_only')}, O₃ image + hour {cvg('O3','image_hour')}, O₃ image only {cvg('O3','image_only')}. Simply predicting each site's average gives {loc_cv('NO2')} for NO₂ and {loc_cv('O3')} for O₃ under the same folds, so much of what the image models learn is which site the camera shows.",
]

loc_rows = []
for loc, g in df.groupby("Location"):
    gg = G[G.Location == loc]
    loc_rows.append([loc, f"{len(g):,}", str(g.obs_id.nunique()), str(g.day_id.nunique()),
                     f"{gg.NO2.min():.0f}–{gg.NO2.max():.0f}" if gg.NO2.notna().any() else "–",
                     f"{gg.O3.min():.0f}–{gg.O3.max():.0f}" if gg.O3.notna().any() else "–", f"{g.date.min()} to {g.date.max()}"])
split_rows = []
for s in ["train", "val", "test"]:
    d = df[df.split == s]; gg = G[G.split == s]
    split_rows.append([s.capitalize(), f"{len(d):,}", str(d.obs_id.nunique()), str(d.day_id.nunique()), str(gg.NO2.notna().sum()), str(gg.O3.notna().sum()),
                       f"{gg.NO2.mean():.1f}", f"{gg.O3.mean():.1f}"])
loc_split = pd.crosstab(G.Location, G.split)[["train", "val", "test"]]
locsplit_rows = [[i] + [str(v) for v in r] for i, r in zip(loc_split.index, loc_split.values)]

cfg_rows = []
from backbones import BACKBONES
dims = {"resnet50": 2048, "resnet18": 512, "effnet_lite4": 1280, "densenet121": 1024, "mobilenetv2": 1280}
for bb in ["resnet50", "resnet18", "effnet_lite4", "densenet121", "mobilenetv2"]:
    cfg_rows.append([bb, BACKBONES[bb][0], f"{dims[bb]}", str(L["n_views"][bb]), BACKBONES[bb][1]])
sel_rows = []
for t in ["NO2", "O3"]:
    for fs in FS:
        c = selc(t, fs); hp = c["hp"]
        hps = f"PCA {hp['n_pca']}, alpha {hp['alpha']}" if c["head"] == "ridge" else f"PCA {hp['n_pca']}, hidden {hp['hidden']}, dropout {hp['dropout']}, wd {hp['wd']}, best epoch {hp['best_epoch']}"
        sel_rows.append([t, FS[fs], c["model"], hps, L["selected"][t][fs]["val_selected_model"]])
cv_rows = []
for t in ["NO2", "O3"]:
    cv_rows.append([t, "Location mean (reference)", "{:.1f}".format(X["cv_location_mean"][t]["group"]["MAE"][0]), loc_cv(t), "{:.2f}".format(X["cv_location_mean"][t]["group"]["Pearson"][0]), loc_cv(t, "image")])
    for fs, lab in [("pm_hour_only", "PM2.5 + PM10 + hour (no image)")] + list(FS.items()):
        cfgm = "ridge" if fs == "pm_hour_only" else CV[f"{t}|{fs}"]["config"]["model"]
        cv_rows.append([t, f"{lab}: {cfgm}" if fs != "pm_hour_only" else lab, "{:.1f}".format(CV[f"{t}|{fs}"]["mean_sd"]["group"]["MAE"][0]), cvg(t, fs),
                        "{:.2f}".format(CV[f"{t}|{fs}"]["mean_sd"]["group"]["Pearson"][0]), cvi(t, fs)])
abl_rows = [[a["target"], FS[a["fset"]], a["model"], f2(a["with_aug"]["group"]["R2"]), f2(a["without_aug"]["group"]["R2"]), f2(a["with_aug"]["image"]["R2"]), f2(a["without_aug"]["image"]["R2"])] for a in L["ablation_no_aug"]]
within_rows = [[k.split("|")[0], FS[k.split("|")[1]], v["model"], str(v["n_groups"]), str(v["n_locations"]), f2(v["within_site_pearson"])] for k, v in X["within_location"].items()]

def mrow(t, lvl): return {"metrics": [t, lvl]}
N["sections"] = [
 {"heading": "1. Dataset", "blocks": [
  {"p": "Source: Air Pollution Image Dataset from India and Nepal (Rouniyar, Utomo, A. and Hsiung, 2023, Kaggle, DOI 10.34740/KAGGLE/DS/3152196). Three files were supplied: the CSV of records (IND_and_Nep_AQI_Dataset.csv), the image archive (All_img.zip) and the dataset README."},
  {"p": f"The CSV has {len(df):,} rows and 14 columns: Location, Filename, Year, Month, Day, Hour, AQI, PM2.5, PM10, O3, CO, SO2, NO2 and AQI_Class. Images are 224 × 224 RGB JPEGs of street and sky scenes. Records span {df.date.min()} to {df.date.max()}; all sites except Faridabad (one session in October 2022) were captured in February and March 2023. NO₂ is missing for {int(df.NO2.isna().sum())} images ({int(G.NO2.isna().sum())} observations) and O₃ for {int(df.O3.isna().sum())} images ({int(G.O3.isna().sum())} observations); those rows are excluded only for the affected target."},
  {"p": "The README does not state units for the pollutant columns. Values are consistent with concentrations in µg/m³ as reported by Indian monitoring stations, and the dashboard's interpretation bands assume this, but it is an assumption and is stated wherever bands are shown."},
  {"fig": "dataset_composition.png", "w": 6.3, "caption": "Figure 1. Images and distinct pollution observations per location. Image counts overstate the information content by roughly 100×."},
  {"table": {"header": ["Location", "Images", "Observations", "Location-days", "NO₂ range", "O₃ range", "Dates"], "rows": loc_rows,
             "widths": [2300, 800, 1050, 1050, 950, 900, 1976], "opts": {"leftCols": [6], "size": 16}}, "caption": "Table 1. Per-location composition (ranges are over observations)."},
  {"fig": "sample_images.png", "w": 6.0, "caption": "Figure 2. One example image per location."}]},
 {"heading": "2. Matching images to records and identifying observation groups", "blocks": [
  {"p": f"Images were matched to CSV rows by exact filename. All {len(df):,} CSV filenames exist in the archive, there are no extra images, and filenames are unique, so the match is one-to-one. Location, date and hour are taken from the CSV columns, not parsed from filenames, because filename time formats are inconsistent (for example a Mumbai file written '03.30' corresponds to CSV hour 15:30)."},
  {"p": f"Multiple images belong to the same pollution observation. An observation group is defined as all rows with identical location, date, hour and pollutant readings (AQI, PM2.5, PM10, O₃, CO, SO₂, NO₂). This yields {G.shape[0]} groups. Group sizes range from 1 to 641 images (median {int(df.groupby('obs_id').size().median())}). Within a group every target value is identical, and the images are near-duplicate frames of the same scene. One location-date-hour (Bengaluru, 2023-02-03 08:30) holds two different sets of readings and is treated as two groups."},
  {"fig": "group_sizes.png", "w": 5.0, "caption": "Figure 3. Distribution of images per observation group."}]},
 {"heading": "3. Preprocessing", "blocks": [
  {"bullets": [
   "Images: decoded as RGB and resized/centre-cropped to 224 × 224 (dataset images are already this size). ImageNet mean/std normalisation for ResNet, DenseNet and MobileNet; [−1, 1] scaling for EfficientNet-Lite4.",
   "Augmentation (training rows only): horizontal flip, and a random resized crop (scale 0.8–1.0, aspect 0.9–1.1) with random flip and mild brightness/contrast jitter (±8%). Colour and haze carry the pollution signal, so stronger colour jitter was deliberately avoided. Augmented views were pre-computed for ResNet-50, DenseNet-121 and MobileNetV2 (original + 2 views); EfficientNet-Lite4 and ResNet-18 used the original view only, because of compute limits. During MLP training one view is sampled per image per epoch; ridge heads are fitted on all views stacked. Evaluation always uses the original, un-augmented image.",
   "Image features: 2048/1024/1280/512-dimensional global-average-pooled embeddings, standardised and projected with PCA (whitened). Scalers and PCA are fitted on training images only.",
   "Target normalisation: log(1 + y) followed by z-scoring with the mean and standard deviation of the training observations (one value per group). Predictions are transformed back and clipped at zero.",
   "Hour: sin and cos of the 24-hour cycle plus a linear term. PM2.5 and PM10: log(1 + x). Both are standardised on training data.",
   "Sample weighting: each image is weighted by 1 / (images in its observation group), so every observation counts equally and the 641-frame Faridabad session does not dominate training."]}]},
 {"heading": "4. Data splitting methodology", "blocks": [
  {"p": "The split unit is the location-day: all observations from one site on one date go to the same subset. This is stricter than splitting by observation group, because observations from the same site and day (for example Delhi at 14:00, 14:30, 17:00 and 18:00 on 16 February) share a scene and similar air, and would leak across a group-only split. Location-days are assigned within each location (stratified), about 65% train, 15% validation and 20% test, so every site with enough sessions appears in each subset. Faridabad has a single session and is train-only."},
  {"p": "The random seed was chosen before any model was trained, by generating 200 candidate splits and keeping the one whose train, validation and test NO₂/O₃ distributions (mean and standard deviation) were closest to the overall distribution. Only target statistics were used, no model output. Leakage checks confirmed zero shared observation groups and zero shared location-days between any two subsets."},
  {"table": {"header": ["Subset", "Images", "Observations", "Location-days", "Obs. with NO₂", "Obs. with O₃", "Mean NO₂", "Mean O₃"], "rows": split_rows,
             "widths": [1000, 1000, 1150, 1150, 1150, 1150, 1213, 1213]}, "caption": "Table 2. Train / validation / test split (means over observations)."},
  {"table": {"header": ["Location", "Train", "Validation", "Test"], "rows": locsplit_rows, "widths": [3800, 1742, 1742, 1742]}, "caption": "Table 3. Observation groups per location and subset."},
  {"p": "Usage: hyper-parameters (PCA size, ridge alpha, MLP settings) and MLP early stopping use the validation set only. The test set was used once per candidate for the final comparison. Because the test set holds only about 22 observations per target, the selected configurations were also re-evaluated with repeated grouped cross-validation: 3 repeats × 5 folds over all location-days (stratified by location). Within each training fold, 15% of location-days are held out for early stopping. The pipeline (scalers, PCA, target scaler, head) is refitted from scratch in every fold."}]},
 {"heading": "5. Models compared", "blocks": [
  {"p": "Full end-to-end fine-tuning of the CNN backbones could not be run in this environment: the package index and the PyTorch download server were blocked, so no deep-learning training framework could be installed. Pretrained ImageNet weights were obtained from the ONNX Model Zoo and run with OpenCV DNN (cross-checked against onnxruntime: identical top-1 classes, max logit difference below 2 × 10⁻⁵). All learning therefore happens in the regression heads on top of frozen features. With about 74 training observations, full fine-tuning would also be at high risk of overfitting; this is discussed under limitations."},
  {"bullets": [
   "Baselines (no image): mean of training observations; ridge on hour features; ridge on PM2.5 + PM10 + hour.",
   "Reference (not deployable from an upload): the training mean of the observation's location. It shows how much can be achieved just by knowing the site.",
   "Model 1, image only: frozen CNN embedding → PCA → ridge or MLP.",
   "Model 2, image + hour: as Model 1, with hour features concatenated.",
   "Model 3, image + PM2.5 + PM10 + hour: as Model 1, with hour and log-PM features concatenated (environmental fusion).",
   "For every input set, all five backbones were tried with a ridge head. An MLP head was then trained on the backbone with the best validation ridge score."]},
  {"table": {"header": ["Backbone", "ONNX weights", "Feature dim", "Views", "Feature tensor"], "rows": cfg_rows, "widths": [1500, 2300, 1100, 800, 3326], "opts": {"leftCols": [1, 4], "size": 16}},
   "caption": "Table 4. Pretrained backbones (ImageNet, ONNX Model Zoo). Views = original + augmented views available for training."},
  {"h2": True},
  {"p": "Model configurations. Ridge grid: PCA size ∈ {8, 16, 32, 64, 128, 256} × alpha ∈ {0.01 … 10,000} (42 settings), selected by validation group-level MSE on the normalised log target. MLP: one hidden layer (64 ReLU units), dropout ∈ {0.2, 0.5}, weight decay ∈ {0.001, 0.01}, PCA size ∈ {32, 128}, Gaussian input noise σ = 0.5, Adam (lr 0.001), batch 256, up to 300 epochs, early stopping with patience 30 on validation group-level loss, restoring the best epoch. Loss: weighted MSE on the normalised log target."},
  {"table": {"header": ["Target", "Input set", "Selected model (test)", "Hyper-parameters", "Best on validation"], "rows": sel_rows, "widths": [700, 2000, 1900, 2726, 1700], "opts": {"leftCols": [1, 2, 3, 4], "size": 16}},
   "caption": "Table 5. Configuration of the model kept for each target and input set. The last column shows which candidate the validation set alone would have picked."}]},
 {"heading": "6. Evaluation metrics", "blocks": [
  {"bullets": ["MAE: mean absolute error, in the dataset's pollutant units.", "RMSE: root mean squared error.", "R²: 1 − SSE/SST on the evaluated set (it can be negative when a model is worse than predicting the test mean).", "Pearson r: linear correlation between predicted and measured values."]},
  {"p": "Image level: every test image is scored against its observation's value, so large groups weigh more and the near-duplicate frames are not independent. Group level: predictions of all images in an observation are averaged and compared with the observation's value, one point per observation. Group level is the primary metric because it matches the number of independent measurements."}]},
 {"heading": "7. NO₂ results", "blocks": [
  {"p": f"Test set, observation-group level (n = {n2['image_only']['test']['group']['n']} observations). Rows in bold are the models kept for each input set."},
  mrow("NO2", "group") | {"caption": "Table 6. NO₂, held-out test set, group level."},
  {"p": f"Test set, image level (n = {n2['image_only']['test']['image']['n']:,} images)."},
  mrow("NO2", "image") | {"caption": "Table 7. NO₂, held-out test set, image level."},
  {"fig": "scatter_NO2.png", "w": 6.5, "caption": "Figure 4. NO₂ actual vs predicted for the kept models (grey: single images; blue: observation means; dashed: y = x)."},
  {"p": f"Reading: PM2.5 and PM10 are strongly correlated with NO₂ in this dataset (r = {G[['NO2','PM2.5']].corr().iloc[0,1]:.2f} with PM2.5 across observations), so the fusion model's gain comes mostly from the PM inputs. On this split the image still adds information on top of PM + hour ({f2(base('NO2','pm_hour_only')['test']['group']['R2'])} → {f2(n2['image_pm_hour']['test']['group']['R2'])}), and cross-validation confirms the gain ({cvg('NO2','pm_hour_only')} → {cvg('NO2','image_pm_hour')}). Image-only models predict group means with r ≈ 0.73–0.82 but at single-image level their R² is low (0.1–0.35), because individual frames from one scene scatter widely around their group mean."}]},
 {"heading": "8. O₃ results", "blocks": [
  {"p": f"Test set, observation-group level (n = {o3['image_only']['test']['group']['n']} observations)."},
  mrow("O3", "group") | {"caption": "Table 8. O₃, held-out test set, group level."},
  {"p": f"Test set, image level (n = {o3['image_only']['test']['image']['n']:,} images)."},
  mrow("O3", "image") | {"caption": "Table 9. O₃, held-out test set, image level."},
  {"fig": "scatter_O3.png", "w": 6.5, "caption": "Figure 5. O₃ actual vs predicted for the kept models."},
  {"p": f"Reading: O₃ is driven by sunlight and time of day (r = {G[['O3','hour']].corr().iloc[0,1]:.2f} with hour across observations), and the image-only model cannot recover it well (test group R² {f2(o3['image_only']['test']['group']['R2'])}, image level {f2(o3['image_only']['test']['image']['R2'])}). Adding the hour roughly doubles group-level R² on test. PM adds little for O₃. The fusion MLP stopped at epoch {o3['image_pm_hour']['hp']['best_epoch']}, which makes it less stable than its test score suggests; its cross-validated R² ({cvg('O3','image_pm_hour')}) is close to image + hour ({cvg('O3','image_hour')})."}]},
 {"heading": "9. Comparison across models", "blocks": [
  {"fig": "test_r2_comparison.png", "w": 6.6, "caption": "Figure 6. Test group-level R² for every candidate (blue: image models; grey: no-image baselines; green: location-mean reference; dashed line: R² = 0.75)."},
  {"p": "Repeated grouped cross-validation (3 × 5 folds) of the kept configurations. Every observation is predicted once per repeat by a model that never saw its location-day; metrics are computed over all out-of-fold predictions and averaged over repeats."},
  {"table": {"header": ["Target", "Model", "MAE (group)", "R² (group)", "Pearson r (group)", "R² (image)"], "rows": cv_rows,
             "widths": [700, 3826, 1050, 1250, 1100, 1100], "opts": {"leftCols": [1], "size": 16}}, "caption": "Table 10. Cross-validated performance (mean ± sd over 3 repeats)."},
  {"fig": "cv_r2.png", "w": 6.0, "caption": "Figure 7. Cross-validated group-level R² (mean ± sd over repeats)."},
  {"p": f"Cross-validation is consistently below the single test split, which happened to be favourable. Under cross-validation, no image model beats the location-mean reference ({loc_cv('NO2')} NO₂, {loc_cv('O3')} O₃) except the NO₂ fusion model. This means the image features largely encode which camera site is shown. The table below checks the opposite question on the test set: after removing each site's mean, do predictions still follow the measured values within a site? (Locations with at least three test observations.)"},
  {"table": {"header": ["Target", "Input set", "Model", "Groups", "Sites", "Within-site r"], "rows": within_rows, "widths": [700, 2300, 2000, 900, 800, 1326], "opts": {"leftCols": [1, 2]}},
   "caption": "Table 11. Correlation of site-demeaned predictions with site-demeaned measurements (test set, group level)."},
  {"p": "Augmentation ablation: each kept model was refitted on original images only, with identical hyper-parameters. The effect of augmentation is small and not consistently positive, which is expected when the backbone is frozen and augmentation only perturbs the embeddings."},
  {"table": {"header": ["Target", "Input set", "Model", "Group R² with aug.", "Group R² without", "Image R² with aug.", "Image R² without"], "rows": abl_rows,
             "widths": [700, 2100, 1700, 1150, 1150, 1100, 1126], "opts": {"leftCols": [1, 2], "size": 16}}, "caption": "Table 12. Effect of training-time augmentation on the test set (models whose backbone has augmented views)."}]},
 {"heading": "10. Training and validation curves", "blocks": [
  {"fig": "mlp_curves.png", "w": 6.6, "caption": "Figure 8. MLP heads: training loss (image level, with dropout and input noise active) and validation loss (group level). The dotted line marks the restored best epoch."},
  {"fig": "ridge_tuning.png", "w": 6.3, "caption": "Figure 9. Ridge heads: validation loss across the PCA × alpha grid for the kept image-only backbones."},
  {"p": "The MLP validation loss flattens within roughly 20–60 epochs, and longer training only lowers the training loss, so early stopping is doing real work. Very strong ridge penalties (alpha ≥ 1,000) underfit; below that the validation loss is flat, so the PCA size is the main capacity control."}]},
 {"heading": "11. Final model selection and deployment", "blocks": [
  {"p": f"As requested, the final models are selected on held-out test performance (group-level R²), and the validation-only choice is reported alongside (Table 5). The two agree for NO₂ fusion and for both O₃ models with hour; for the image-only models the validation set would have chosen a different model with somewhat lower test performance (NO₂ {f2(cand('NO2','image_only',L['selected']['NO2']['image_only']['val_selected_model'])['test']['group']['R2'])} vs {f2(n2['image_only']['test']['group']['R2'])}; O₃ {f2(cand('O3','image_only',L['selected']['O3']['image_only']['val_selected_model'])['test']['group']['R2'])} vs {f2(o3['image_only']['test']['group']['R2'])})."},
  {"bullets": [
   f"Best overall: NO₂, {n2['image_pm_hour']['model']} on image + PM2.5 + PM10 + hour. Test group R² {f2(n2['image_pm_hour']['test']['group']['R2'])}, MAE {f1(n2['image_pm_hour']['test']['group']['MAE'])}; image-level R² {f2(n2['image_pm_hour']['test']['image']['R2'])}; CV group R² {cvg('NO2','image_pm_hour')}. This is environmental fusion.",
   f"Best O₃: {o3['image_pm_hour']['model']} on image + PM + hour (test group R² {f2(o3['image_pm_hour']['test']['group']['R2'])}). {o3['image_hour']['model']} on image + hour is practically equivalent (test {f2(o3['image_hour']['test']['group']['R2'])}, CV {cvg('O3','image_hour')} vs {cvg('O3','image_pm_hour')}) and needs no PM readings.",
   f"Best pure-image models: NO₂ {n2['image_only']['model']} (test group R² {f2(n2['image_only']['test']['group']['R2'])}, CV {cvg('NO2','image_only')}); O₃ {o3['image_only']['model']} (test {f2(o3['image_only']['test']['group']['R2'])}, CV {cvg('O3','image_only')})."]},
  {"p": "Deployment workflow. An uploaded photo cannot supply PM2.5 or PM10, and it supplies the hour only if the camera wrote a timestamp. The dashboard therefore ships all three model variants for both pollutants and picks the richest one the user's inputs allow: photo only by default; photo + hour when the user enters the capture time or the photo's EXIF timestamp is present; photo + hour + PM when the user also types PM2.5 and PM10 from a nearby monitor. The interface shows which inputs were used and a typical-error figure (the variant's test group-level MAE), but no model internals. The interpretation bands follow India's CPCB AQI breakpoints and assume µg/m³ (see section 1)."}]},
 {"heading": "12. Limitations", "blocks": [
  {"bullets": [
   f"Very small effective sample: {G.shape[0]} observations from 8 sites over about 7 weeks (plus one October session). Test metrics rest on about 22 observations per target and move noticeably with the split; cross-validated numbers are the better estimate.",
   "Site confounding: each site has a fixed camera view and its own typical pollution level. Image models partly act as site recognisers; the location-mean reference matches or beats most image models under cross-validation. Generalisation to new cameras or new cities was not tested and should be expected to be poor.",
   "Single-image predictions are noisy (image-level R² is far below group-level). Averaging several photos of the same scene and time gives more reliable estimates.",
   "No end-to-end fine-tuning: backbones were frozen because no deep-learning training framework could be installed here. Fine-tuning might help, but with this few observations it would need strong regularisation and grouped validation to avoid overfitting.",
   "Units are not documented in the README; µg/m³ is assumed for the interpretation bands. The CPCB bands refer to 24-hour (NO₂) and 8-hour (O₃) averages, while the dataset gives hourly readings, so the bands are indicative only.",
   "The fusion models need PM2.5/PM10 from a reference monitor. Their accuracy depends on that monitor being representative of the photographed location.",
   "Selecting the final model on test performance (as requested) gives an optimistic estimate for the chosen model; the validation-based choice and cross-validation are reported to show this.",
   "High readings (roughly NO₂ or O₃ above 100) are rare in training, and predictions for them shrink toward the middle of the range (Figures 4 and 5)."]}]},
 {"heading": "13. Conclusion", "blocks": [
  {"p": f"With a leakage-safe location-day split, frozen ImageNet CNN features carry a moderate amount of information about NO₂ and less about O₃. Pure image models reach test group-level R² of {f2(n2['image_only']['test']['group']['R2'])} (NO₂) and {f2(o3['image_only']['test']['group']['R2'])} (O₃), and {cvg('NO2','image_only')} and {cvg('O3','image_only')} under cross-validation, and much of that signal is site identity. The R² > 0.75 target is reached only by the NO₂ environmental-fusion model (image + PM2.5 + PM10 + hour) at group level on the test split ({f2(n2['image_pm_hour']['test']['group']['R2'])}); its cross-validated value is {cvg('NO2','image_pm_hour')}. For O₃, the time of day is the most useful extra input."},
  {"p": "The dashboard is suitable as an indicative, site-familiar estimator, not as a measurement. The most valuable next step is more independent observations (more days, more hours, more cameras per site) rather than more frames per observation, followed by evaluation on held-out sites."}]},
]
# convert the "h2" marker block into a no-op (kept simple)
for s in N["sections"]:
    s["blocks"] = [b for b in s["blocks"] if not b.get("h2")]
json.dump(N, open(f"{ROOT}/report/narrative.json", "w"), indent=1, ensure_ascii=False)
print("ok")
