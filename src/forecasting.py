"""Core data preparation, forecasting, evaluation, and visualization functions."""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TIMESTAMP = "timestamp"
TARGET = "energy_kwh"
FEATURES = [
    "hour", "dayofweek", "month", "is_weekend",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_mean_168",
]
MIN_HOURS = 24 * 8


def generate_synthetic_data(days: int = 180, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible hourly building demand for a software demo."""
    if days * 24 < MIN_HOURS:
        raise ValueError("Use at least 8 days so weekly lag features and a test set can be created.")
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    hour = index.hour.to_numpy()
    weekday = index.dayofweek.to_numpy()
    # Baseline + morning/evening activity + weekday occupancy + gradual drift.
    daily = 2.2 + 1.1 * np.sin((hour - 7) * 2 * np.pi / 24)
    morning_peak = 2.4 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
    evening_peak = 3.0 * np.exp(-0.5 * ((hour - 19) / 2.8) ** 2)
    occupancy = np.where(weekday < 5, 1.8, 0.6)
    drift = np.linspace(0, 0.8, len(index))
    noise = rng.normal(0, 0.45, len(index))
    values = np.maximum(0.3, 7.0 + daily + morning_peak + evening_peak + occupancy + drift + noise)
    return pd.DataFrame({TIMESTAMP: index, TARGET: values.round(3)})


def load_and_clean_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load a CSV or create simulated data; return continuous hourly data."""
    if csv_path is None:
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(csv_path)
        required = {TIMESTAMP, TARGET}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; missing: {sorted(missing)}"
            )
    df = df[[TIMESTAMP, TARGET]].copy()
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP, TARGET])
    df = df[df[TARGET] >= 0]
    df = df.sort_values(TIMESTAMP).drop_duplicates(TIMESTAMP, keep="last")
    df = df.set_index(TIMESTAMP).resample("h").mean()
    df[TARGET] = df[TARGET].interpolate(method="time").ffill().bfill()
    df = df.reset_index()
    if len(df) < MIN_HOURS:
        raise ValueError("At least 8 days of hourly data are required for training and evaluation.")
    if df[TARGET].isna().any():
        raise ValueError("Energy values still contain missing data after cleaning.")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create causal time and lag features; target at time t is predicted from earlier data."""
    out = df.copy().sort_values(TIMESTAMP).set_index(TIMESTAMP)
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["lag_1"] = out[TARGET].shift(1)
    out["lag_24"] = out[TARGET].shift(24)
    out["lag_168"] = out[TARGET].shift(168)
    # shift(1) prevents the current target from leaking into rolling features.
    out["rolling_mean_24"] = out[TARGET].shift(1).rolling(24).mean()
    out["rolling_mean_168"] = out[TARGET].shift(1).rolling(168).mean()
    out = out.dropna(subset=FEATURES + [TARGET]).reset_index()
    return out


def train_and_evaluate(featured: pd.DataFrame, test_fraction: float = 0.2):
    """Chronological holdout evaluation; no random shuffle."""
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("test_fraction should be between 0.1 and 0.4.")
    split = int(len(featured) * (1 - test_fraction))
    train, test = featured.iloc[:split], featured.iloc[split:]
    model = RandomForestRegressor(
        n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])
    metrics = {
        "mae_kwh": float(mean_absolute_error(test[TARGET], pred)),
        "rmse_kwh": float(np.sqrt(mean_squared_error(test[TARGET], pred))),
        "r2": float(r2_score(test[TARGET], pred)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_start": str(test[TIMESTAMP].iloc[0]),
        "test_end": str(test[TIMESTAMP].iloc[-1]),
        "model": "RandomForestRegressor",
    }
    predictions = pd.DataFrame({
        TIMESTAMP: test[TIMESTAMP].to_numpy(),
        "actual_kwh": test[TARGET].to_numpy(),
        "predicted_kwh": pred,
    })
    return model, metrics, predictions


def forecast_next_hours(history: pd.DataFrame, model, hours: int = 24) -> pd.DataFrame:
    """Recursive multi-step forecast; each new prediction feeds later lag features."""
    if hours < 1 or hours > 168:
        raise ValueError("hours must be between 1 and 168.")
    hist = history[[TIMESTAMP, TARGET]].copy().sort_values(TIMESTAMP)
    hist[TIMESTAMP] = pd.to_datetime(hist[TIMESTAMP])
    series = hist.set_index(TIMESTAMP)[TARGET].astype(float).asfreq("h")
    series = series.interpolate(method="time").ffill().bfill()
    if len(series) < 168:
        raise ValueError("Need at least 168 hourly records for weekly lag features.")
    forecasts = []
    for _ in range(hours):
        next_time = series.index[-1] + pd.Timedelta(hours=1)
        vals = series.to_numpy()
        row = {
            "hour": next_time.hour,
            "dayofweek": next_time.dayofweek,
            "month": next_time.month,
            "is_weekend": int(next_time.dayofweek >= 5),
            "lag_1": vals[-1],
            "lag_24": vals[-24],
            "lag_168": vals[-168],
            "rolling_mean_24": float(np.mean(vals[-24:])),
            "rolling_mean_168": float(np.mean(vals[-168:])),
        }
        prediction = max(0.0, float(model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
        forecasts.append({TIMESTAMP: next_time, "predicted_kwh": prediction})
        series.loc[next_time] = prediction
    return pd.DataFrame(forecasts)


def save_artifacts(history, model, metrics, predictions, future, out_dir="outputs", model_dir="models"):
    out_path, model_path = Path(out_dir), Path(model_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_path / "cleaned_energy.csv", index=False)
    predictions.to_csv(out_path / "test_predictions.csv", index=False)
    future.to_csv(out_path / "next_24_hours_forecast.csv", index=False)
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(
        {"model": model, "features": FEATURES, "history": history},
        model_path / "energy_forecast_model.joblib",
    )

    plt.figure(figsize=(12, 4))
    plt.plot(pd.to_datetime(history[TIMESTAMP]), history[TARGET], linewidth=0.8)
    plt.title("Hourly Energy Consumption (input data)")
    plt.xlabel("Timestamp"); plt.ylabel("Energy (kWh)"); plt.tight_layout()
    plt.savefig(out_path / "energy_history.png", dpi=150); plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(predictions[TIMESTAMP], predictions["actual_kwh"], label="Actual", linewidth=1)
    plt.plot(predictions[TIMESTAMP], predictions["predicted_kwh"], label="Predicted", linewidth=1)
    plt.title("Actual vs Predicted Energy Consumption (chronological test set)")
    plt.xlabel("Timestamp"); plt.ylabel("Energy (kWh)"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path / "actual_vs_predicted.png", dpi=150); plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(future[TIMESTAMP], future["predicted_kwh"], marker="o", markersize=3)
    plt.title("Next 24 Hours: Forecast Energy Consumption")
    plt.xlabel("Timestamp"); plt.ylabel("Predicted energy (kWh)"); plt.xticks(rotation=30)
    plt.tight_layout(); plt.savefig(out_path / "next_24_hours.png", dpi=150); plt.close()

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance"); plt.tight_layout()
    plt.savefig(out_path / "feature_importance.png", dpi=150); plt.close()
