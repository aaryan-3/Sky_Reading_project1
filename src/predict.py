"""Inference for the trained NO2 / O3 models.

    from predict import Predictor
    p = Predictor("models")                       # folder with config.json, *.npz and backbone .onnx files
    p.predict("photo.jpg")                        # image-only model
    p.predict("photo.jpg", hour=14.5)             # image + hour model (if bundled)
    p.predict("photo.jpg", hour=14.5, pm25=120, pm10=180)   # image + PM2.5 + PM10 + hour model

CLI:  python predict.py models photo.jpg [--hour 14.5] [--pm25 120 --pm10 180]
"""
import os, sys, json, argparse, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import Backbone, resize_center
from data import hour_features

def _load_rgb_bytes(data):
    arr = np.frombuffer(data, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("The file is not a readable image (use JPG or PNG).")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return resize_center(im, 224)  # training images are 224x224; other sizes are resized + centre-cropped

class _Variant:
    def __init__(self, path, cfg):
        z = np.load(path)
        self.cfg = cfg
        self.m, self.s, self.P, self.ps = z["m"], z["s"], z["P"], z["ps"]
        self.tm, self.ts = z["tm"], z["ts"]
        self.mu, self.sd = float(z["y_mu"]), float(z["y_sd"])
        if cfg["head"] == "ridge":
            self.coef, self.b = z["coef"], float(z["b"][0])
        else:
            self.W1, self.b1, self.W2, self.b2 = z["W1"], z["b1"], z["W2"], z["b2"]

    def predict(self, feat, tab):
        Z = (feat.astype(np.float64) - self.m) / self.s
        if self.P.size: Z = (Z @ self.P) / self.ps
        X = Z if tab.shape[1] == 0 else np.concatenate([Z, (tab - self.tm) / self.ts], 1)
        if self.cfg["head"] == "ridge":
            z = X @ self.coef + self.b
        else:
            z = (np.maximum(X @ self.W1 + self.b1, 0) @ self.W2 + self.b2)[:, 0]
        return np.expm1(z * self.sd + self.mu).clip(min=0)

class Predictor:
    def __init__(self, model_dir):
        self.dir = model_dir
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.variants = {}
        self.backbones = {}
        for target, vs in self.cfg["variants"].items():
            for fset, vc in vs.items():
                self.variants[(target, fset)] = _Variant(os.path.join(model_dir, vc["file"]), vc)
                if vc["backbone"] not in self.backbones:
                    self.backbones[vc["backbone"]] = Backbone(vc["backbone"], model_dir)

    def available_inputs(self):
        return sorted({f for _, f in self.variants})

    def predict(self, image, hour=None, pm25=None, pm10=None):
        data = open(image, "rb").read() if isinstance(image, str) else image
        rgb = _load_rgb_bytes(data)[None]
        feats = {}
        if pm25 is not None and pm10 is not None and hour is not None:
            fset = "image_pm_hour"; tab = np.concatenate([hour_features([hour]), np.log1p([[pm25, pm10]])], 1)
        elif hour is not None:
            fset = "image_hour"; tab = hour_features([hour])
        else:
            fset = "image_only"; tab = np.zeros((1, 0))
        out = {"inputs_used": fset}
        for target in self.cfg["variants"]:
            fs = fset if (target, fset) in self.variants else "image_only"
            v = self.variants[(target, fs)]
            bb = v.cfg["backbone"]
            if bb not in feats: feats[bb] = self.backbones[bb].features(rgb)
            t = tab if fs == fset else np.zeros((1, 0))
            out[target] = float(v.predict(feats[bb], t)[0])
            out[f"{target}_model"] = fs
            out[f"{target}_typical_error"] = v.cfg.get("typical_error")
        return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("image")
    ap.add_argument("--hour", type=float); ap.add_argument("--pm25", type=float); ap.add_argument("--pm10", type=float)
    a = ap.parse_args()
    print(json.dumps(Predictor(a.model_dir).predict(a.image, a.hour, a.pm25, a.pm10), indent=1))
