import os
import logging
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA
from darts.metrics import rmse
from hyperopt import fmin, tpe, hp, Trials
from matplotlib import pyplot as plt

# Function to prepare time series
def prepare_time_series(df, target, freq='H'):
    # Ensure 'ut_ms' is numeric
    df['ut_ms'] = pd.to_numeric(df['ut_ms'], errors='coerce')  # Convert to numeric; invalid values become NaN
    df = df.dropna(subset=['ut_ms'])  # Drop rows with NaN in 'ut_ms'

    # Convert 'ut_ms' to datetime
    df['timestamp'] = pd.to_datetime(df['ut_ms'], unit='ms', errors='coerce')  # Convert milliseconds to datetime
    df = df.dropna(subset=['timestamp'])  # Drop rows with invalid datetime

    # Check and print the dataframe
    print("DataFrame after timestamp conversion:")
    print(df[['timestamp', target]].head())  # Print a subset for verification

    # Set 'timestamp' as index and create the time series
    df = df.set_index('timestamp')
    ts = TimeSeries.from_dataframe(df, value_cols=[target], freq=freq)
    return ts

# Function for ARIMA hyperparameter tuning
def tune_arima(ts, forecast_horizon):
    def objective(params):
        p, d, q = params
        model = ARIMA(p=p, d=d, q=q)
        model.fit(ts)
        forecast = model.predict(n=forecast_horizon)
        error = rmse(ts[-forecast_horizon:], forecast)
        return error

    space = [
        hp.randint('p', 5),
        hp.randint('d', 2),
        hp.randint('q', 5)
    ]
    
    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    return best_params

# Function for ARIMA forecasting
def forecast_arima(ts, forecast_horizon, order):
    model = ARIMA(p=order[0], d=order[1], q=order[2])
    model.fit(ts)
    forecast = model.predict(n=forecast_horizon)
    return forecast

# Function to plot forecast comparison
def plot_forecast_comparison(actual_forecast, forecasts_dict, run_dir):
    plt.figure(figsize=(10, 6))
    for label, forecast in forecasts_dict.items():
        forecast.plot(label=label)
    actual_forecast.plot(label="Actual", color='black')
    plt.legend()
    plt.title("Forecast Comparison")
    plt.savefig(os.path.join(run_dir, "forecast_comparison.png"))
    plt.show()

# Save summary report
def save_summary_report(metrics, filepath):
    with open(filepath, 'w') as f:
        f.write("Model Metrics and Parameters Summary\n")
        for model, metric in metrics.items():
            f.write(f"{model} RMSE: {metric:.4f}\n")

# Main execution function
def main():
    logging.basicConfig(filename='forecasting.log', level=logging.INFO)
    print("\n--- Starting Forecasting ---")
    logging.info("--- Starting Forecasting ---")
    
    try:
        # Load your data
        df = pd.read_csv('/Users/rajatpundir/Desktop/Mars/runs/run_20250118_020608_baf06fc4-e8d6-4715-b65f-f578896d39ba/cleaned_power_consumption.csv')
        ts = prepare_time_series(df, target='NPWD2372')  # Adjust target column

        forecast_horizon = 30  # Forecast next 30 days
        
        # Tune and forecast ARIMA
        print("\n--- Tuning ARIMA ---")
        best_arima_order = tune_arima(ts, forecast_horizon)
        print("\n--- Forecasting with Tuned ARIMA ---")
        arima_forecast = forecast_arima(ts, forecast_horizon, order=best_arima_order)
        
        # Compute forecasting errors
        actual_forecast = ts[-forecast_horizon:]
        arima_rmse = rmse(actual_forecast, arima_forecast)
        
        # Print RMSE values
        print("\nForecasting Evaluation Metrics (RMSE):")
        print(f"ARIMA RMSE: {arima_rmse:.4f}")
        logging.info(f"ARIMA RMSE: {arima_rmse:.4f}")
        
        # Plot forecast comparison
        forecasts_dict = {"ARIMA": arima_forecast}
        run_dir = '/Users/rajatpundir/Desktop/Mars/runs/run_20250118_020608_baf06fc4-e8d6-4715-b65f-f578896d39ba'
        plot_forecast_comparison(actual_forecast, forecasts_dict, run_dir)

        # Save Summary Report
        summary_report_path = os.path.join(run_dir, 'prediction_summary_report.txt')
        save_summary_report(metrics={"ARIMA": arima_rmse}, filepath=summary_report_path)

        print(f"Summary report saved to: {summary_report_path}")
        logging.info(f"Summary report saved to: {summary_report_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()
