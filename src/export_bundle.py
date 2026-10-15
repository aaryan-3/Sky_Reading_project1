"""Assemble the deployable model bundle: config.json + per-target/input-set .npz + required backbone .onnx."""
import os, sys, json, glob, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import BACKBONES

def main(model_dir, weights_dir, deploy=None):
    variants = {}
    for js in sorted(glob.glob(os.path.join(model_dir, "*_image_*.json"))):
        meta = json.load(open(js)); t, fs = meta["target"], meta["feature_set"]
        if deploy and fs not in deploy: continue
        variants.setdefault(t, {})[fs] = {"file": os.path.basename(js).replace(".json", ".npz"), "backbone": meta["backbone"], "head": meta["head"],
                                         "typical_error": round(meta["test"]["group"]["MAE"], 1)}
    need = {v["backbone"] for vs in variants.values() for v in vs.values()}
    for bb in need:
        f = BACKBONES[bb][0]
        if not os.path.exists(os.path.join(model_dir, f)): shutil.copy(os.path.join(weights_dir, f), os.path.join(model_dir, f))
    cfg = {"targets": ["NO2", "O3"], "units": "dataset units (assumed ug/m3)", "variants": variants}
    json.dump(cfg, open(os.path.join(model_dir, "config.json"), "w"), indent=1)
    print(json.dumps(cfg, indent=1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(",") if len(sys.argv) > 3 else None)
