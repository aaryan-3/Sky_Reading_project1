"""Air-quality-from-a-photo dashboard (Flask backend).
Run:  python dashboard/app.py   then open http://127.0.0.1:8000
"""
import os, sys, io
from flask import Flask, request, jsonify, send_from_directory
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from predict import Predictor

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HERE, "..", "models"))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
predictor = Predictor(MODEL_DIR)

# India CPCB AQI breakpoints (sub-index bands). Assumes the model output is in ug/m3, which the
# dataset README does not state explicitly - see the evaluation report.
BANDS = {
    "NO2": [(40, "Good"), (80, "Satisfactory"), (180, "Moderate"), (280, "Poor"), (400, "Very poor"), (float("inf"), "Severe")],
    "O3": [(50, "Good"), (100, "Satisfactory"), (168, "Moderate"), (208, "Poor"), (748, "Very poor"), (float("inf"), "Severe")],
}
ADVICE = {
    "Good": "Minimal impact.",
    "Satisfactory": "Minor breathing discomfort for sensitive people.",
    "Moderate": "Breathing discomfort for people with lung or heart conditions, children and older adults.",
    "Poor": "Breathing discomfort for most people on prolonged exposure.",
    "Very poor": "Respiratory illness on prolonged exposure.",
    "Severe": "Affects healthy people; serious impact on those with existing disease.",
}
ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]

def band(target, v):
    for hi, name in BANDS[target]:
        if v <= hi: return name

def exif_hour(data):
    """Hour of day from the photo's EXIF DateTimeOriginal/DateTime, if present."""
    try:
        from PIL import Image
        ex = Image.open(io.BytesIO(data)).getexif()
        dt = ex.get_ifd(0x8769).get(36867) or ex.get(306)
        if dt:
            hh, mm = dt.split(" ")[1].split(":")[:2]
            return int(hh) + int(mm) / 60.0
    except Exception:
        pass
    return None

def _num(name):
    v = request.form.get(name, "").strip()
    if not v: return None
    x = float(v)
    if x < 0: raise ValueError(f"{name} cannot be negative")
    return x

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/api/info")
def info():
    return jsonify({"inputs": predictor.available_inputs(), "scales": {t: [[None if b[0] == float("inf") else b[0], b[1]] for b in v] for t, v in BANDS.items()}})

@app.post("/api/predict")
def predict():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify({"error": "Choose a photo to analyse."}), 400
    try:
        hour, pm25, pm10 = _num("hour"), _num("pm25"), _num("pm10")
        if hour is not None and not (0 <= hour < 24): raise ValueError("Time must be between 00:00 and 23:59.")
        if (pm25 is None) != (pm10 is None): raise ValueError("Enter both PM2.5 and PM10, or leave both empty.")
        data = f.read(); hour_src = "entered" if hour is not None else None
        if hour is None:
            hour = exif_hour(data); hour_src = "photo" if hour is not None else None
        if pm25 is not None and hour is None: raise ValueError("Add the time the photo was taken to use PM readings.")
        r = predictor.predict(data, hour=hour, pm25=pm25, pm10=pm10)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    out = {"readings": {}}
    for t in ["NO2", "O3"]:
        b = band(t, r[t])
        out["readings"][t] = {"value": round(r[t], 1), "band": b, "advice": ADVICE[b], "level": ORDER.index(b), "typical_error": r.get(f"{t}_typical_error")}
    worst = max(out["readings"].items(), key=lambda kv: kv[1]["level"])
    if worst[1]["level"] <= 1:
        s = "The air in this scene looks fairly clean for nitrogen dioxide and ozone."
    else:
        s = f"{'Nitrogen dioxide' if worst[0] == 'NO2' else 'Ozone'} is the main concern here: {worst[1]['band'].lower()} level."
    out["summary"] = s
    out["time_source"] = hour_src
    out["used"] = {"image_only": "photo only", "image_hour": "photo and time", "image_pm_hour": "photo, time and PM readings"}[r["inputs_used"]]
    return jsonify(out)

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)), debug=False)
