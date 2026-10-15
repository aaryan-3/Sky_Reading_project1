"""Pretrained ImageNet CNN backbones (ONNX model zoo) used as feature extractors.
OpenCV DNN is used so that the global-average-pooled penultimate layer can be read directly."""
import cv2, numpy as np, os

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

BACKBONES = {
    # name: (onnx file, feature layer, layout, preprocessing)
    "resnet50": ("resnet50-v1-7.onnx", "resnetv17_pool1_fwd", "nchw", "imagenet"),
    "resnet18": ("resnet18-v1-7.onnx", "resnetv15_pool1_fwd", "nchw", "imagenet"),
    "effnet_lite4": ("efficientnet-lite4-11.onnx", "efficientnet-lite4/model/head/Squeeze:0", "nhwc", "pm1"),
    "densenet121": ("densenet-12.onnx", "pool5_1", "nchw", "imagenet"),
    "mobilenetv2": ("mobilenetv2-12.onnx", "472", "nchw", "imagenet"),
}

def preprocess(rgb_batch, kind, layout):
    """rgb_batch: (N,224,224,3) uint8 RGB"""
    x = rgb_batch.astype(np.float32)
    if kind == "imagenet":
        x = (x / 255.0 - MEAN) / STD
    else:  # efficientnet-lite: [-1,1]
        x = (x - 127.0) / 128.0
    if layout == "nchw":
        x = x.transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x)

class Backbone:
    def __init__(self, name, weights_dir):
        f, layer, layout, kind = BACKBONES[name]
        self.name, self.layout, self.kind = name, layout, kind
        self.net = cv2.dnn.readNetFromONNX(os.path.join(weights_dir, f))
        self.layer = layer or self._find_pool()

    def _find_pool(self):
        names = self.net.getLayerNames()
        for n in reversed(names):
            lt = self.net.getLayer(n).type
            if "Pool" in lt or "pool" in n.lower():
                return n
        raise RuntimeError("no pooling layer")

    def features(self, rgb_batch):
        x = preprocess(rgb_batch, self.kind, self.layout)
        self.net.setInput(x)
        out = self.net.forward(self.layer)
        return out.reshape(out.shape[0], -1).astype(np.float32)

def load_rgb(path, size=224):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"cannot read image {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[:2] != (size, size):
        im = resize_center(im, size)
    return im

def resize_center(im, size=224):
    h, w = im.shape[:2]
    s = size / min(h, w)
    im = cv2.resize(im, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return im[y:y + size, x:x + size]
