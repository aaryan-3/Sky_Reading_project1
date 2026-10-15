"""Regression heads on top of frozen CNN embeddings (NumPy only).
- FeaturePipeline: standardise -> PCA (fit on training images only) -> whiten; tabular features standardised.
- RidgeHead: sample-weighted closed-form ridge regression.
- MLPHead: 1-hidden-layer MLP trained with Adam, dropout, input-noise augmentation, weight decay and
  early stopping on a group-held-out validation set (group-level MSE)."""
import numpy as np

class TargetScaler:
    """log1p then z-score, fitted on training observations (one value per observation group)."""
    def fit(self, y_group):
        z = np.log1p(y_group); self.mu, self.sd = float(np.nanmean(z)), float(np.nanstd(z)); return self
    def transform(self, y): return (np.log1p(y) - self.mu) / self.sd
    def inverse(self, z): return np.expm1(z * self.sd + self.mu).clip(min=0)
    def state(self): return {"mu": self.mu, "sd": self.sd}

class FeaturePipeline:
    def __init__(self, n_pca=128):
        self.n_pca = n_pca
    def fit(self, X_img, X_tab):
        X = X_img.astype(np.float64)
        self.m = X.mean(0); self.s = X.std(0) + 1e-6
        Z = (X - self.m) / self.s
        if self.n_pca and self.n_pca < Z.shape[1]:
            # PCA via SVD on a subsample for speed
            idx = np.random.default_rng(0).choice(len(Z), min(len(Z), 6000), replace=False)
            _, S, Vt = np.linalg.svd(Z[idx], full_matrices=False)
            self.P = Vt[: self.n_pca].T
            self.ps = (S[: self.n_pca] / np.sqrt(len(idx))) + 1e-6
        else:
            self.P = None
        self.tm = X_tab.mean(0) if X_tab.shape[1] else np.zeros(0)
        self.ts = X_tab.std(0) + 1e-6 if X_tab.shape[1] else np.zeros(0)
        return self
    def transform(self, X_img, X_tab):
        parts = []
        if X_img is not None and X_img.shape[1]:
            Z = (X_img.astype(np.float64) - self.m) / self.s
            if self.P is not None: Z = (Z @ self.P) / self.ps
            parts.append(Z)
        if X_tab.shape[1]: parts.append((X_tab - self.tm) / self.ts)
        return np.concatenate(parts, 1).astype(np.float32)
    def state(self):
        return dict(m=self.m, s=self.s, P=self.P if self.P is not None else np.zeros(0), ps=self.ps if self.P is not None else np.zeros(0),
                    tm=self.tm, ts=self.ts)

class RidgeHead:
    def __init__(self, alpha=10.0): self.alpha = alpha
    def fit(self, X, y, w):
        W = w / w.sum()
        xm = (W[:, None] * X).sum(0); ym = (W * y).sum()
        Xc = X - xm; A = Xc.T @ (W[:, None] * Xc) * len(X) + self.alpha * np.eye(X.shape[1])
        self.coef = np.linalg.solve(A, Xc.T @ (W * (y - ym)) * len(X)); self.b = ym - xm @ self.coef
        return self
    def predict(self, X): return X @ self.coef + self.b
    def state(self): return {"coef": self.coef, "b": np.array([self.b])}

def group_mean(pred, groups):
    u, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, pred); c = np.bincount(inv)
    return u, s / c, inv

class MLPHead:
    def __init__(self, hidden=128, dropout=0.3, wd=1e-3, lr=1e-3, noise=0.3, epochs=300, patience=25, batch=256, seed=0):
        self.__dict__.update(dict(hidden=hidden, dropout=dropout, wd=wd, lr=lr, noise=noise, epochs=epochs,
                                  patience=patience, batch=batch, seed=seed))
    def _init(self, d):
        r = np.random.default_rng(self.seed)
        self.W1 = (r.standard_normal((d, self.hidden)) * np.sqrt(2 / d)).astype(np.float32); self.b1 = np.zeros(self.hidden, np.float32)
        self.W2 = (r.standard_normal((self.hidden, 1)) * np.sqrt(1 / self.hidden)).astype(np.float32); self.b2 = np.zeros(1, np.float32)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.mom = [np.zeros_like(p) for p in self.params]; self.vel = [np.zeros_like(p) for p in self.params]; self.t = 0
    def _forward(self, X, train=False, rng=None):
        if train and self.noise: X = X + rng.standard_normal(X.shape).astype(np.float32) * self.noise
        h = X @ self.W1 + self.b1; a = np.maximum(h, 0)
        if train and self.dropout:
            m = (rng.random(a.shape) > self.dropout).astype(np.float32) / (1 - self.dropout); a = a * m
        else: m = None
        return X, h, a, m, (a @ self.W2 + self.b2)[:, 0]
    def predict(self, X): return self._forward(X.astype(np.float32))[-1]
    def fit(self, X_views, y, w, val=None):
        """X_views: list of arrays (augmented views of the same training rows); one view sampled per row per epoch.
        val: (X_val, y_val_z, groups_val) evaluated at group level for early stopping."""
        rng = np.random.default_rng(self.seed); n, d = X_views[0].shape; self._init(d)
        XV = np.stack(X_views).astype(np.float32)
        w = (w / w.mean()).astype(np.float32); y = y.astype(np.float32)
        best, bad, self.history = np.inf, 0, []
        for ep in range(self.epochs):
            vsel = rng.integers(0, len(X_views), n); perm = rng.permutation(n); tl = 0.0
            for i in range(0, n, self.batch):
                b = perm[i:i + self.batch]
                Xb = XV[vsel[b], b]
                Xn, h, a, m, out = self._forward(Xb, True, rng)
                err = (out - y[b]) * w[b]; tl += float((err * (out - y[b])).sum())
                g = (2 * err / len(b))[:, None]
                gW2 = a.T @ g; gb2 = g.sum(0)
                ga = g @ self.W2.T
                if m is not None: ga = ga * m
                gh = ga * (h > 0)
                gW1 = Xn.T @ gh; gb1 = gh.sum(0)
                grads = [gW1 + self.wd * self.W1, gb1, gW2 + self.wd * self.W2, gb2]
                self.t += 1
                for k, (p, gr) in enumerate(zip(self.params, grads)):
                    self.mom[k] = 0.9 * self.mom[k] + 0.1 * gr; self.vel[k] = 0.999 * self.vel[k] + 0.001 * gr * gr
                    mh = self.mom[k] / (1 - 0.9 ** self.t); vh = self.vel[k] / (1 - 0.999 ** self.t)
                    p -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            rec = {"epoch": ep + 1, "train_loss": tl / n}
            if val is not None:
                Xv, yv, gv = val
                pv = self.predict(Xv); u, pg, inv = group_mean(pv, gv); _, yg, _ = group_mean(yv, gv)
                vl = float(np.mean((pg - yg) ** 2)); rec["val_loss"] = vl
                rec["val_img_loss"] = float(np.mean((pv - yv) ** 2))
                if vl < best - 1e-4:
                    best, bad = vl, 0; self.best_epoch = ep + 1
                    self.best = [p.copy() for p in self.params]
                else:
                    bad += 1
            self.history.append(rec)
            if val is not None and bad >= self.patience: break
        if val is not None:
            for p, bp in zip(self.params, self.best): p[...] = bp
        return self
    def state(self): return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(y); y, p = y[ok], p[ok]
    err = p - y
    ss = ((y - y.mean()) ** 2).sum()
    r = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 and np.std(y) > 0 else float("nan")
    return {"n": int(len(y)), "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": float(1 - (err ** 2).sum() / ss), "Pearson": float(r)}
