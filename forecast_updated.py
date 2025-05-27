import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import psutil
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Utility: Memory Usage -----
def get_memory_usage():
    """Return a string summarizing system memory usage."""
    mem = psutil.virtual_memory()
    return (f"Total: {mem.total/1e9:.2f}GB | "
            f"Available: {mem.available/1e9:.2f}GB | "
            f"Used: {mem.used/1e9:.2f}GB | "
            f"Percent: {mem.percent}%")

# ----- Power Data Loading -----
def load_and_preprocess_power_csv(filepath):
    """
    Reads a single CSV, checks for 'ut_ms' or 'utb_ms' columns,
    converts them to a datetime index, and sorts by index.
    """
    df = pd.read_csv(filepath, engine="python")
    if "ut_ms" in df.columns:
        df["ut_ms"] = pd.to_datetime(df["ut_ms"].astype(float), unit="ms", errors="coerce")
        df.set_index("ut_ms", inplace=True)
    elif "utb_ms" in df.columns:
        df["utb_ms"] = pd.to_datetime(df["utb_ms"].astype(float), unit="ms", errors="coerce")
        df.set_index("utb_ms", inplace=True)
    
    df.sort_index(inplace=True)
    return df

def load_all_power_data(folder_path):
    """
    Scans the specified folder for CSV files whose names contain 'power--',
    loads them, and concatenates into a single DataFrame sorted by datetime index.
    """
    # Filter only CSV files with "power--" in the name
    all_files = [f for f in os.listdir(folder_path)
                 if f.endswith(".csv") and "power--" in f.lower()]
    
    all_dfs = []
    for file in all_files:
        filepath = os.path.join(folder_path, file)
        try:
            df = load_and_preprocess_power_csv(filepath)
            all_dfs.append(df)
            logging.info(f"Loaded power file: {file}, rows={len(df)}")
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")

    if not all_dfs:
        return pd.DataFrame()  # Return empty if no files found
    
    merged_power = pd.concat(all_dfs, axis=0)
    merged_power.sort_index(inplace=True)
    return merged_power

def fill_gaps(df, col="NPWD2372"):
    """Forward fills missing values in the specified column."""
    df[col] = df[col].ffill()
    return df

# ----- Forecasting -----
def forecast_arima(df, target="NPWD2372", horizon=30):
    """
    Uses pmdarima's auto_arima to forecast power consumption.
    The forecast horizon is adjusted based on data frequency.
    Returns a DataFrame with columns ['Date', 'Forecast'].
    """
    series = df[target].dropna()
    if len(series) < 10:
        st.warning("Not enough data points for forecasting.")
        return pd.DataFrame(columns=["Date", "Forecast"])
    
    # Fit auto_arima model
    model = pm.auto_arima(series, start_p=0, start_q=0,
                          max_p=5, max_q=5, d=None,
                          seasonal=False, stepwise=True,
                          suppress_warnings=True, error_action="ignore")
    
    freq = pd.infer_freq(series.index) or "D"
    if freq == "H":
        periods = horizon * 24
    elif freq in ["T", "min"]:
        periods = horizon * 24 * 60
    else:
        periods = horizon

    forecast_vals = model.predict(n_periods=periods)
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
    fc_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast_vals})
    return fc_df

# ----- Anomaly Detection -----
def detect_anomalies_arima(df, target="NPWD2372", order=(2,1,2), threshold_multiplier=3):
    """
    Fits an ARIMA model on the target series,
    computes residuals, and flags anomalies where:
      |residual| > threshold_multiplier * std.
    Returns a DataFrame with columns: actual, predicted, residual, anomaly.
    """
    series = df[target].dropna()
    if len(series) < 10:
        st.warning("Not enough data points for anomaly detection.")
        return pd.DataFrame()
    
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    pred = model_fit.predict(start=series.index[0], end=series.index[-1])
    residuals = series - pred
    threshold = threshold_multiplier * residuals.std()
    anomaly_flags = (residuals.abs() > threshold).astype(int)
    
    anom_df = pd.DataFrame({
        "actual": series,
        "predicted": pred,
        "residual": residuals,
        "anomaly": anomaly_flags
    }, index=series.index)
    return anom_df

def compute_severity(anom_df):
    """
    Computes severity for anomalies (Low, Medium, High) using the 33rd and 66th percentiles of residuals.
    """
    anom_detected = anom_df[anom_df["anomaly"] == 1].copy()
    if anom_detected.empty:
        anom_df["severity"] = np.nan
        return anom_df
    
    resid_abs = anom_detected["residual"].abs()
    q33, q66 = np.percentile(resid_abs, [33, 66])
    
    def get_sev(val):
        if val < q33:
            return "Low"
        elif val < q66:
            return "Medium"
        else:
            return "High"
    
    severity_series = resid_abs.apply(get_sev)
    anom_df["severity"] = "N/A"
    anom_df.loc[severity_series.index, "severity"] = severity_series
    return anom_df

# ----- Event Data -----
def classify_event_file(filename, df):
    """
    Classifies the file based on its name or (optionally) content.
    Recognized: ftl, emop, evtf, saaf, dmop, ltdata
    """
    filename_lower = filename.lower()
    if "ftl" in filename_lower:
        return "Failure Log"
    elif "emop" in filename_lower:
        return "Emergency Operations"
    elif "evtf" in filename_lower:
        return "Event Tracking"
    elif "saaf" in filename_lower:
        return "Safety Alerts"
    elif "dmop" in filename_lower:
        return "DMOP Events"
    elif "ltdata" in filename_lower:
        return "Telemetry Data"
    
    # Optional fallback if there's a "description" column
    if "description" in df.columns:
        desc_lower = df["description"].astype(str).str.lower()
        if desc_lower.str.contains("failure").any():
            return "Failure Log"
        elif desc_lower.str.contains("emergency").any():
            return "Emergency Operations"
        elif desc_lower.str.contains("tracking").any():
            return "Event Tracking"
        elif desc_lower.str.contains("safety").any():
            return "Safety Alerts"
    
    return "Unknown Event Type"

def load_context_events(folder_path):
    """
    Loads all CSV files in the folder that do NOT contain 'power--' in the filename,
    classifies them, merges them, and returns a single DataFrame sorted by timestamp.
    """
    event_files = [f for f in os.listdir(folder_path)
                   if f.endswith(".csv") and "power--" not in f.lower()]
    
    all_events = []
    for file in event_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path, engine="python")
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            continue

        event_type = classify_event_file(file, df)
        
        # Convert timestamp
        if "ut_ms" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ut_ms"].astype(float), unit="ms", errors="coerce")
        elif "utb_ms" in df.columns:
            df["timestamp"] = pd.to_datetime(df["utb_ms"].astype(float), unit="ms", errors="coerce")
        else:
            time_cols = [col for col in df.columns if "time" in col.lower()]
            if time_cols:
                df["timestamp"] = pd.to_datetime(df[time_cols[0]], errors="coerce")
        
        df.sort_values("timestamp", inplace=True)
        df["event_type"] = event_type
        all_events.append(df)
    
    if all_events:
        merged_events = pd.concat(all_events, ignore_index=True)
        merged_events.sort_values("timestamp", inplace=True)
        return merged_events
    else:
        return pd.DataFrame()

def fast_merge_anomalies_events(anom_df, events_df, tolerance="1H"):
    """
    Uses a nearest merge to associate each anomaly with the closest event (within a tolerance).
    """
    anom_df = anom_df.reset_index().rename(columns={'ut_ms': 'timestamp'})
    events_df = events_df.copy().sort_values("timestamp")

    if "description" in events_df.columns:
        merge_cols = ["timestamp", "event_type", "description"]
    else:
        merge_cols = ["timestamp", "event_type"]

    merged = pd.merge_asof(
        anom_df, events_df[merge_cols],
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance)
    )
    
    merged["related_event"] = merged.get("description", "No Event").fillna("No Event")
    merged["event_category"] = merged["event_type"].fillna("Unknown Event")
    merged.drop(columns=["event_type"], inplace=True)
    merged = merged.set_index("timestamp")
    return merged

# ----- Main Streamlit App -----
def main():
    st.title("Mars Express Power Consumption Dashboard")
    st.write("Memory usage: " + get_memory_usage())
    
    # Hardcode absolute paths for train_set and test_set
    TRAIN_PATH = "/Users/rajatpundir/Desktop/Mars/train_set"
    TEST_PATH = "/Users/rajatpundir/Desktop/Mars/test_set"
    
    # Let user choose train_set or test_set
    phase = st.selectbox("Select Dataset Phase", ["train_set", "test_set"])
    
    if phase == "train_set":
        folder_path = TRAIN_PATH
    else:
        folder_path = TEST_PATH
    
    st.write(f"Loading power data from: {folder_path}")
    
    # Load all power CSV files in the folder
    df_power = load_all_power_data(folder_path)
    
    if df_power.empty:
        st.error(f"No power files found in {folder_path} (filenames must contain 'power--').")
        return
    
    st.write("Power Data Preview (first 10 rows):")
    st.dataframe(df_power.head(10))
    
    # Create two tabs for Forecasting and Anomaly Detection
    tab1, tab2 = st.tabs(["Forecasting", "Anomaly Detection"])
    
    # ----- FORECASTING TAB -----
    with tab1:
        st.header("Forecasting")
        horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30)
        
        if st.button("Run Forecast"):
            fc_df = forecast_arima(df_power, target="NPWD2372", horizon=horizon)
            if not fc_df.empty:
                st.write("Forecast Results (first 10 rows):")
                st.dataframe(fc_df.head(10))
                
                # Plot historical data + forecast
                df_plot = df_power.reset_index()
                x_col = "ut_ms" if "ut_ms" in df_plot.columns else df_plot.columns[0]
                fig = px.line(df_plot, x=x_col, y="NPWD2372", title="Historical Power Consumption")
                fig.add_scatter(x=fc_df["Date"], y=fc_df["Forecast"], mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)
                
                # Download the forecast as CSV
                csv_data = fc_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Forecast CSV", csv_data, "forecast.csv", "text/csv")
            else:
                st.warning("Forecasting returned empty results. Check if there's enough data.")
    
    # ----- ANOMALY DETECTION TAB -----
    with tab2:
        st.header("Anomaly Detection")
        
        # Option to fill missing NPWD2372 values
        if st.checkbox("Fill missing values in 'NPWD2372'?"):
            df_power = fill_gaps(df_power, "NPWD2372")
            st.write("Missing values after fill:", df_power["NPWD2372"].isna().sum())
        
        st.subheader("ARIMA Parameters & Threshold")
        p = st.number_input("ARIMA p", min_value=0, max_value=10, value=2, step=1)
        d = st.number_input("ARIMA d", min_value=0, max_value=5, value=1, step=1)
        q = st.number_input("ARIMA q", min_value=0, max_value=10, value=2, step=1)
        order = (p, d, q)
        
        thresh_mult = st.slider("Threshold Multiplier", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        
        if st.button("Detect Anomalies"):
            anom_results = detect_anomalies_arima(df_power, target="NPWD2372",
                                                  order=order,
                                                  threshold_multiplier=thresh_mult)
            if anom_results.empty:
                st.warning("No anomalies detected or not enough data.")
            else:
                # Compute severity
                anom_results = compute_severity(anom_results)
                # Filter to only anomalies
                anom_detected = anom_results[anom_results["anomaly"] == 1]
                st.write(f"Detected {anom_detected.shape[0]} anomalies.")
                
                # Plot anomalies
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(anom_results.index, anom_results["actual"], label="Actual")
                anomaly_idx = anom_results.index[anom_results["anomaly"] == 1]
                ax2.scatter(anomaly_idx, anom_results.loc[anomaly_idx, "actual"],
                            color="red", label="Anomaly")
                ax2.set_title("Anomaly Detection (ARIMA Residuals)")
                ax2.legend()
                st.pyplot(fig2)
                
                # Load event files from the same folder (if any)
                st.write(f"Loading event files from: {folder_path}")
                df_events = load_context_events(folder_path)
                
                if df_events.empty:
                    st.warning("No event files found or loaded.")
                    anom_detected["related_event"] = "No Event"
                    anom_detected["event_category"] = "No Event"
                else:
                    st.write("Merged Events Data Preview (first 10 rows):")
                    st.dataframe(df_events.head(10))
                    # Merge anomalies with events
                    anom_detected = fast_merge_anomalies_events(anom_detected, df_events, tolerance="1H")
                
                st.write("Anomaly Results (first 20 rows):")
                st.dataframe(anom_detected.head(20))
                
                # Download anomaly results
                csv_out = anom_detected.to_csv().encode("utf-8")
                st.download_button("Download Anomaly CSV", csv_out, "anomalies.csv", "text/csv")
        
        st.write("Memory usage: " + get_memory_usage())

if __name__ == "__main__":
    main()
