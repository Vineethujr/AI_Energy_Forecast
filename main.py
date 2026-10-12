"""Run the end-to-end energy forecasting pipeline."""
import argparse
import json
from src.forecasting import (
    load_and_clean_data, make_features, train_and_evaluate,
    forecast_next_hours, save_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description="Hourly energy forecasting demo")
    parser.add_argument("--data", default=None, help="Optional CSV with timestamp,energy_kwh")
    parser.add_argument("--hours", type=int, default=24, help="Forecast horizon (1–168 hours)")
    args = parser.parse_args()

    history = load_and_clean_data(args.data)
    featured = make_features(history)
    model, metrics, predictions = train_and_evaluate(featured)
    future = forecast_next_hours(history, model, args.hours)
    save_artifacts(history, model, metrics, predictions, future)

    print("\n=== Energy Forecasting Results ===")
    print(f"Rows after cleaning: {len(history)}")
    print(f"Training rows: {metrics['train_rows']}")
    print(f"Testing rows: {metrics['test_rows']}")
    print(f"MAE:  {metrics['mae_kwh']:.3f} kWh")
    print(f"RMSE: {metrics['rmse_kwh']:.3f} kWh")
    print(f"R²:   {metrics['r2']:.3f}")
    print(f"\nForecast for next {len(future)} hours:")
    print(future.to_string(index=False))
    print("\nSaved model to models/energy_forecast_model.joblib")
    print("Saved CSV, metrics and plots to outputs/")


if __name__ == "__main__":
    main()
