"""Download the pretrained ImageNet ONNX backbones (ONNX Model Zoo) into the model folder.
Usage: python src/download_weights.py [models] [--all]   (--all also fetches backbones only needed for retraining)"""
import os, sys, json, shutil, subprocess, urllib.request, urllib.error
BASE = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/"
FILES = {"resnet50-v1-7.onnx": "resnet/model/", "resnet18-v1-7.onnx": "resnet/model/",
         "efficientnet-lite4-11.onnx": "efficientnet-lite4/model/", "densenet-12.onnx": "densenet-121/model/",
         "mobilenetv2-12.onnx": "mobilenet/model/"}
SIZES = {"resnet50-v1-7.onnx": 102583340, "resnet18-v1-7.onnx": 46820737, "efficientnet-lite4-11.onnx": 51946641,
         "densenet-12.onnx": 32726877, "mobilenetv2-12.onnx": 13964571}

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backbones import BACKBONES
    if "--all" in sys.argv:
        need = set(FILES)
    else:
        cfg = json.load(open(os.path.join(d, "config.json")))
        need = {BACKBONES[v["backbone"]][0] for vs in cfg["variants"].values() for v in vs.values()}
    for f in sorted(need):
        p = os.path.join(d, f)
        if os.path.exists(p) and os.path.getsize(p) == SIZES[f]:
            print("ok", f); continue
        print("downloading", f, f"({SIZES[f] / 1e6:.0f} MB)", flush=True)
        url = BASE + FILES[f] + f
        try:
            urllib.request.urlretrieve(url, p + ".part")
        except urllib.error.URLError as e:
            # python.org builds on macOS often lack CA certificates; curl uses the system trust store
            if shutil.which("curl") is None: raise
            print("  Python could not verify the certificate (" + str(e.reason)[:60] + "...), retrying with curl", flush=True)
            subprocess.run(["curl", "-fL", "--retry", "3", "-o", p + ".part", url], check=True)
        if os.path.getsize(p + ".part") != SIZES[f]:
            raise SystemExit(f"{f}: unexpected size, download incomplete")
        os.replace(p + ".part", p)
    print("done")

if __name__ == "__main__":
    main()
