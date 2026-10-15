"""Dataset loading, image-to-record matching, grouping and leakage-safe splitting."""
import numpy as np, pandas as pd, os

TARGETS = ["NO2", "O3"]
OBS_COLS = ["Location", "date", "Hour", "AQI", "PM2.5", "PM10", "O3", "CO", "SO2", "NO2"]

def hour_float(h):
    hh, mm = str(h).split(":")
    return int(hh) + int(mm) / 60.0

def load(csv, imgdir=None):
    df = pd.read_csv(csv)
    if imgdir is not None:
        files = set(os.listdir(imgdir))
        missing = (~df.Filename.isin(files)).sum()
        assert missing == 0, f"{missing} CSV rows have no image"
    df["date"] = df.Year.astype(str) + "-" + df.Month.astype(str).str.zfill(2) + "-" + df.Day.astype(str).str.zfill(2)
    key = df[OBS_COLS].astype(object).where(df[OBS_COLS].notna(), "NA").astype(str).agg("|".join, axis=1)
    df["obs_id"] = pd.factorize(key)[0]           # one pollution observation (identical readings)
    df["day_id"] = pd.factorize(df.Location + "|" + df.date)[0]  # location-day session (split unit)
    df["hour"] = df.Hour.map(hour_float)
    return df

def make_split(df, test_frac=0.2, val_frac=0.15, seed=59):
    """Split by location-day sessions, stratified by location so every location with
    enough sessions appears in train, validation and test. All images of an observation
    (and of any observation from the same location-day) go to exactly one subset."""
    rng = np.random.default_rng(seed)
    split = {}
    for loc, g in df.groupby("Location"):
        days = g.day_id.unique().copy()
        rng.shuffle(days)
        n = len(days)
        n_test = 0 if test_frac == 0 else (int(round(n * test_frac)) if n >= 3 else (1 if n == 2 else 0))
        n_val = int(round(n * val_frac)) if n >= 4 else 0
        for i, d in enumerate(days):
            split[d] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return df.day_id.map(split)

def hour_features(h):
    h = np.asarray(h, float)
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), (h - 12) / 4], 1)

def tab_features(df, kind):
    """kind: 'none' | 'hour' | 'pm_hour'"""
    if kind == "none":
        return np.zeros((len(df), 0), np.float32)
    H = hour_features(df.hour.values)
    if kind == "hour":
        return H.astype(np.float32)
    P = np.log1p(df[["PM2.5", "PM10"]].values.astype(float))
    return np.concatenate([H, P], 1).astype(np.float32)
