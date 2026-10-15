"""Minimal protobuf wire-format reader for ONNX graphs (the `onnx` package is not installable here).
Used only to list node output tensor names so a feature tensor can be requested from OpenCV DNN."""
def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s; s += 7
        if c < 0x80: return r, i

def fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i); f, wt = key >> 3, key & 7
        if wt == 0: v, i = _varint(b, i)
        elif wt == 1: v = b[i:i+8]; i += 8
        elif wt == 5: v = b[i:i+4]; i += 4
        elif wt == 2:
            l, i = _varint(b, i); v = b[i:i+l]; i += l
        else: raise ValueError(wt)
        yield f, wt, v

def nodes(path):
    b = open(path, 'rb').read()
    graph = next(v for f, wt, v in fields(b) if f == 7)
    out = []
    for f, wt, v in fields(graph):
        if f == 1:
            d = {"in": [], "out": [], "name": "", "op": ""}
            for g, _, w in fields(v):
                if g == 1: d["in"].append(bytes(w).decode())
                elif g == 2: d["out"].append(bytes(w).decode())
                elif g == 3: d["name"] = bytes(w).decode()
                elif g == 4: d["op"] = bytes(w).decode()
            out.append(d)
    return out
