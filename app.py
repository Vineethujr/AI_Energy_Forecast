"""Optional local Flask prediction API. Run main.py first to train/save the model."""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from src.forecasting import TARGET, TIMESTAMP, FEATURES

app = Flask(__name__)
ARTIFACT_PATH = "models/energy_forecast_model.joblib"


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        if "timestamp" not in payload:
            return jsonify({"error": "JSON field 'timestamp' is required"}), 400
        target_time = pd.Timestamp(payload["timestamp"])
        artifact = joblib.load(ARTIFACT_PATH)
        history = artifact["history"].copy()
        history[TIMESTAMP] = pd.to_datetime(history[TIMESTAMP])
        series = history.set_index(TIMESTAMP)[TARGET].asfreq("h").interpolate(method="time").ffill().bfill()

        # This endpoint supports the next timestamp only, so lag inputs are known.
        if target_time != series.index[-1] + pd.Timedelta(hours=1):
            return jsonify({
                "error": "For this demo API, timestamp must be exactly one hour after the latest training record.",
                "latest_training_timestamp": str(series.index[-1]),
            }), 400
        vals = series.to_numpy()
        row = {
            "hour": target_time.hour,
            "dayofweek": target_time.dayofweek,
            "month": target_time.month,
            "is_weekend": int(target_time.dayofweek >= 5),
            "lag_1": vals[-1], "lag_24": vals[-24], "lag_168": vals[-168],
            "rolling_mean_24": float(series.iloc[-24:].mean()),
            "rolling_mean_168": float(series.iloc[-168:].mean()),
        }
        prediction = max(0.0, float(artifact["model"].predict(pd.DataFrame([row], columns=FEATURES))[0]))
        return jsonify({"timestamp": str(target_time), "predicted_energy_kwh": round(prediction, 3)})
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Run `python main.py` first."}), 500
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
