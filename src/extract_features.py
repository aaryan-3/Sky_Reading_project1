"""Extract frozen pretrained-CNN embeddings for every image (one row per CSV record, CSV order).
View 0 = original image, view 'flip' = horizontal flip, views 'augK' = random resized crop (scale 0.8-1.0)
+ horizontal flip + mild brightness/contrast jitter (+-8%). Colour/haze is the signal, so jitter is kept small."""
import sys, os, time, numpy as np, pandas as pd, cv2
from backbones import Backbone, load_rgb

def augment(im, rng):
    h, w = im.shape[:2]
    s = rng.uniform(0.8, 1.0); ar = rng.uniform(0.9, 1.1)
    ch, cw = int(h * s / np.sqrt(ar)), int(w * s * np.sqrt(ar)); ch, cw = min(ch, h), min(cw, w)
    y, x = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    im = cv2.resize(im[y:y+ch, x:x+cw], (224, 224), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5: im = im[:, ::-1]
    a = rng.uniform(0.92, 1.08); b = rng.uniform(-0.08, 0.08) * 255 * 0.25
    im = np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return im

def main(backbone, view, csv, imgdir, out, bs=32):
    df = pd.read_csv(csv)
    net = Backbone(backbone, os.environ.get('WEIGHTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights')))
    seed = {"orig": 0, "flip": 1}[view] if view in ("orig", "flip") else 100 + int(view.replace("aug", "") or 0)
    rng = np.random.default_rng(seed)
    feats = []; t = time.time()
    for i in range(0, len(df), bs):
        ims = []
        for f in df.Filename.iloc[i:i+bs]:
            im = load_rgb(os.path.join(imgdir, f))
            if view == "flip": im = im[:, ::-1]
            elif view.startswith("aug"): im = augment(im, rng)
            ims.append(np.ascontiguousarray(im))
        feats.append(net.features(np.stack(ims)))
        if i % (bs * 50) == 0: print(backbone, view, i, f"{time.time()-t:.0f}s", flush=True)
    F = np.concatenate(feats).astype(np.float16)
    np.save(out, F); print("saved", out, F.shape, f"{time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    main(*sys.argv[1:5], sys.argv[5])
