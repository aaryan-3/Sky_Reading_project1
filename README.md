# Sky Reading: NO₂ and O₃ from outdoor photos

Computer-vision models that estimate nitrogen dioxide (NO₂) and ozone (O₃) from street/sky photos,
trained on the *Air Pollution Image Dataset from India and Nepal* (Rouniyar et al., 2023), plus a
web dashboard and a model-evaluation report.

## Headline results (held-out test set, honest numbers)

| Target | Inputs | Model | Test R² (group) | Test R² (image) | CV R² (group) |
|---|---|---|---|---|---|
| NO₂ | image only | EfficientNet-Lite4 + ridge | 0.53 | 0.11 | 0.33 ± 0.04 |
| NO₂ | image + hour | EfficientNet-Lite4 + ridge | 0.51 | 0.03 | 0.27 ± 0.02 |
| NO₂ | image + PM2.5 + PM10 + hour | DenseNet-121 + ridge | **0.80** | 0.62 | 0.51 ± 0.03 |
| O₃ | image only | ResNet-50 + ridge | 0.26 | −0.03 | 0.21 ± 0.10 |
| O₃ | image + hour | DenseNet-121 + MLP | 0.51 | 0.33 | 0.28 ± 0.02 |
| O₃ | image + PM2.5 + PM10 + hour | DenseNet-121 + MLP | 0.54 | 0.21 | 0.29 ± 0.04 |

*Group* = mean prediction over all images of one pollution observation (the independent unit).
*CV* = 3 × 5-fold grouped cross-validation over all observations, the more reliable estimate.
R² > 0.75 is reached only by the NO₂ **environmental-fusion** model (image + PM + hour), only at group
level, only on the single test split. The best **pure-image** results are 0.53 (NO₂) and 0.26 (O₃).
Full details and caveats are in `report/Model_Evaluation_Report.docx`.

## Key facts about the data

* 12,240 images, all matched one-to-one to CSV rows by filename.
* They belong to only **116 pollution observations** (same location, date, hour and readings),
  1 to 641 images each. Every image of an observation has identical targets.
* Split unit: **location-day** (stricter than observation), stratified by location:
  64 / 15 / 20 location-days → 74 / 18 / 24 observations → 7,494 / 1,903 / 2,843 images. No overlap.

## Layout

```
dashboard/        Flask app (app.py) + single-page frontend (static/index.html)
models/           Deployable bundle: config.json, 6 trained head files (*.npz + *.json metadata);
                  the 3 ONNX backbones are fetched by src/download_weights.py
src/              backbones.py, data.py, models.py, extract_features.py, experiment.py,
                  extra_analysis.py, predict.py, export_bundle.py, make_figures.py, write_narrative.py, onnx_min.py
results/          results.json (all candidates, val/test metrics, tuning, MLP histories),
                  cv.json, extra_analysis.json, test_predictions.csv, experiment.log
report/           Model_Evaluation_Report.docx, figures/, build_report.js, narrative.json
requirements.txt
```

## Run the dashboard

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/download_weights.py                           # fetches the 3 pretrained backbones (~187 MB) into models/
python dashboard/app.py                                  # → http://127.0.0.1:8000
```

Upload a photo, then press **Estimate air quality**. Which model variant runs depends on what is known:

| What you provide | Model used |
|---|---|
| Photo only | image-only model |
| Photo + time (typed in, or read automatically from the photo's EXIF timestamp) | image + hour model |
| Photo + time + PM2.5 + PM10 (from a nearby monitor) | image + PM + hour model (best NO₂) |

Environment variables: `MODEL_DIR` (default `models/`), `HOST`, `PORT`.
API: `POST /api/predict` (multipart: `image`, optional `hour` as decimal hours, `pm25`, `pm10`).

Interpretation bands follow India's CPCB AQI breakpoints and assume the dataset values are µg/m³
(the dataset README does not state units). Treat outputs as indicative, not as measurements.

## Command-line inference

```bash
python src/predict.py models photo.jpg
python src/predict.py models photo.jpg --hour 14.5
python src/predict.py models photo.jpg --hour 14.5 --pm25 120 --pm10 180
```

```python
import sys; sys.path.insert(0, "src")
from predict import Predictor
p = Predictor("models")
p.predict("photo.jpg", hour=14.5)   # {'NO2': ..., 'O3': ..., 'inputs_used': 'image_hour', ...}
```

## Reproduce training

Pretrained ImageNet weights come from the ONNX Model Zoo
(`https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/...`):
`resnet50-v1-7.onnx`, `resnet18-v1-7.onnx`, `efficientnet-lite4-11.onnx`, `densenet-12.onnx`, `mobilenetv2-12.onnx`.
Put them in a `weights/` folder, the CSV at `data/IND_and_Nep_AQI_Dataset.csv` and the images in
`data/All_img/` (or point `DATA_CSV`, `IMG_DIR`, `FEATS` and `PROJ` elsewhere).

```bash
export WEIGHTS_DIR=weights
# 1. embeddings (orig for all backbones; flip + aug1 for resnet50, densenet121, mobilenetv2)
python src/extract_features.py resnet50 orig  data/IND_and_Nep_AQI_Dataset.csv data/All_img feats/resnet50_orig.npy
python src/extract_features.py resnet50 flip  ...  feats/resnet50_flip.npy
python src/extract_features.py resnet50 aug1  ...  feats/resnet50_aug1.npy
# ... repeat for the other backbones
# 2. model selection, test evaluation, cross-validation, model export
FEATS=feats PROJ=. python src/experiment.py resnet50,effnet_lite4,densenet121,resnet18,mobilenetv2
python src/extra_analysis.py
python src/export_bundle.py models weights
# 3. report
python src/make_figures.py && python src/write_narrative.py && node report/build_report.js
```

CPU only: feature extraction takes ~3–20 min per backbone and view; training + CV ~10 min.

## Notes and limitations

* Backbones are **frozen** (no end-to-end fine-tuning): PyTorch/TensorFlow could not be installed in
  the build environment, so features come from ONNX models via OpenCV DNN and the heads (ridge, MLP
  with Adam/dropout/early stopping) are implemented in NumPy.
* With 116 observations from 8 fixed cameras, image models partly learn *which site* is shown.
  Under cross-validation, a site-average reference (not available from an upload) matches or beats
  every pure-image model. Expect poor transfer to new cameras or cities.
* Single-image predictions are noisy; several photos of the same scene give a steadier estimate.
