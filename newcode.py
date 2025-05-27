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

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- Utility: Memory Usage -----
def get_memory_usage():
    mem = psutil.virtual_memory()
    return f"Total: {mem.total/1e9:.2f}GB | Available: {mem.available/1e9:.2f}GB | Used: {mem.used/1e9:.2f}GB | Percent: {mem.percent}%"

# ----- Data Loading & Preprocessing -----
def load_power_data(uploaded_file):
    """
    Loads a CSV file containing power data.
    Expects a 'ut_ms' column for timestamps.
    """
    df = pd.read_csv(uploaded_file, engine="python")
    df['ut_ms'] = pd.to_datetime(df['ut_ms'].astype(float), unit='ms', errors='coerce')
    df.set_index('ut_ms', inplace=True)
    df.sort_index(inplace=True)
    return df

def fill_gaps(df, col="NPWD2372"):
    """Forward-fills missing values in the specified column."""
    df[col] = df[col].ffill()
    return df

# ----- Forecasting Functions -----
def forecast_arima(df, target="NPWD2372", horizon=30):
    """
    Uses pmdarima's auto_arima to forecast the target column.
    For speed, max_p and max_q are limited to 2.
    Returns a DataFrame with columns ['Date', 'Forecast'].
    """
    series = df[target].dropna()
    if len(series) < 10:
        st.warning("Not enough data points for forecasting.")
        return pd.DataFrame(columns=["Date", "Forecast"])
    with st.spinner("Fitting ARIMA model for forecasting..."):
        model = pm.auto_arima(
            series,
            start_p=0, start_q=0,
            max_p=2, max_q=2,
            d=None,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )
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
    return pd.DataFrame({"Date": future_dates, "Forecast": forecast_vals})

# ----- Anomaly Detection Functions -----
def detect_anomalies_arima(df, target="NPWD2372", order=(2,1,2), threshold_multiplier=0.2):
    """
    Fits an ARIMA model on the target column,
    computes residuals, and flags anomalies when:
      |residual| > (mean + threshold_multiplier * std).
    Returns a DataFrame with columns: actual, predicted, residual, anomaly.
    """
    series = df[target].dropna()
    if len(series) < 10:
        st.warning("Not enough data points for anomaly detection.")
        return pd.DataFrame()
    with st.spinner("Fitting ARIMA model for anomaly detection..."):
        model = ARIMA(series, order=order).fit()
    pred = model.predict(start=series.index[0], end=series.index[-1])
    residuals = series - pred
    threshold = residuals.mean() + threshold_multiplier * residuals.std()
    anomaly_flags = (residuals.abs() > threshold).astype(int)
    return pd.DataFrame({
        "actual": series,
        "predicted": pred,
        "residual": residuals,
        "anomaly": anomaly_flags
    }, index=series.index)

def compute_severity(anom_df):
    """
    Computes severity for anomalies based on the 33rd and 66th percentiles of absolute residuals.
    Labels anomalies as Low, Medium, or High.
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

# ----- Event Data Functions -----
def classify_event_description(desc):
    """
    Classifies an event based on the event_description text.
    Recognized keywords (case-insensitive):
      ftl, emop, evtf, saaf, dmop, ltdata, solar flare, communication fault.
    If none are found, returns 'unexplained'.
    """
    desc_lower = str(desc).lower()
    if "ftl" in desc_lower:
        return "ftl"
    elif "emop" in desc_lower:
        return "emop"
    elif "evtf" in desc_lower:
        return "evtf"
    elif "saaf" in desc_lower:
        return "saaf"
    elif "dmop" in desc_lower:
        return "dmop"
    elif "ltdata" in desc_lower:
        return "ltdata"
    elif "solar flare" in desc_lower:
        return "solar flare"
    elif "communication fault" in desc_lower:
        return "communication fault"
    else:
        return "unexplained"

def load_context_events(uploaded_file):
    """
    Loads an events CSV from an uploaded file.
    Expects a timestamp column (ut_ms, utb_ms, or any column with 'time'),
    and an 'event_description' column.
    Then, it classifies each row based on the content of event_description.
    """
    df = pd.read_csv(uploaded_file, engine="python")
    if "ut_ms" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ut_ms"], unit="ms", errors="coerce")
    elif "utb_ms" in df.columns:
        df["timestamp"] = pd.to_datetime(df["utb_ms"], unit="ms", errors="coerce")
    else:
        time_cols = [col for col in df.columns if "time" in col.lower()]
        if time_cols:
            df["timestamp"] = pd.to_datetime(df[time_cols[0]], errors="coerce")
    df.sort_values("timestamp", inplace=True)
    if "event_description" not in df.columns:
        df["event_description"] = "No Description"
    df["event_type"] = df["event_description"].apply(classify_event_description)
    return df

def fast_merge_anomalies_events(anom_df, events_df, rounding="1H"):
    """
    Rounds anomaly and event timestamps to the specified interval,
    then merges them. The merged DataFrame gets:
      - 'related_event' from event_description
      - 'event_category' from event_type
    """
    anom_df = anom_df.copy()
    anom_df["anomaly_rounded"] = anom_df.index.round(rounding)
    
    events_df = events_df.copy()
    events_df["event_rounded"] = events_df["timestamp"].round(rounding)
    
    merged = pd.merge(
        anom_df.reset_index(),
        events_df[["event_rounded", "event_description", "event_type"]],
        left_on="anomaly_rounded",
        right_on="event_rounded",
        how="left"
    )
    merged["related_event"] = merged["event_description"].fillna("No Event")
    merged["event_category"] = merged["event_type"].fillna("unexplained")
    for col in ["event_description", "event_rounded", "anomaly_rounded", "event_type"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)
    merged.set_index("ut_ms", inplace=True)
    return merged

# ----- Main Dashboard -----
def main():
    st.title("Mars Express Dashboard")
    st.write("Memory usage: " + get_memory_usage())
    
    # Two tabs: Forecasting and Anomaly Detection
    tab1, tab2 = st.tabs(["Forecasting", "Anomaly Detection"])
    
    # ------------- FORECASTING TAB -------------
    with tab1:
        st.header("Forecasting")
        uploaded_fc = st.file_uploader("Upload CLEANED power data CSV", type=["csv"], key="fc")
        horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30)
        if uploaded_fc is not None:
            df_fc = load_power_data(uploaded_fc)
            st.write("Data Preview (Power Data):")
            st.dataframe(df_fc.head())
            st.write("Columns:", df_fc.columns.tolist())
            st.write("Missing Values:", df_fc.isna().sum())
            if st.checkbox("Fill missing values (NPWD2372)?", key="fill_fc"):
                df_fc = fill_gaps(df_fc, "NPWD2372")
                st.write("After filling:", df_fc.isna().sum())
            st.write("Memory usage: " + get_memory_usage())
            if st.button("Run Forecast", key="run_fc"):
                fc_df = forecast_arima(df_fc, "NPWD2372", horizon)
                if not fc_df.empty:
                    st.write("Forecast Results:")
                    st.dataframe(fc_df.head(10))
                    df_plot = df_fc.reset_index()
                    fig = px.line(df_plot, x="ut_ms", y="NPWD2372", title="Historical Power Consumption")
                    fig.add_scatter(x=fc_df["Date"], y=fc_df["Forecast"], mode="lines", name="Forecast")
                    st.plotly_chart(fig, use_container_width=True)
                    csv_data = fc_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Forecast CSV", csv_data, "forecast.csv", "text/csv")
                else:
                    st.error("Forecasting failed.")
    
    # ------------- ANOMALY DETECTION TAB -------------
    with tab2:
        st.header("Anomaly Detection")
        uploaded_anom = st.file_uploader("Upload RAW power data CSV", type=["csv"], key="anom")
        uploaded_events = st.file_uploader("Upload EVENTS CSV (optional)", type=["csv"], key="events")
        
        # Provide a button to download a sample events file for testing
        if st.button("Download Sample Events CSV"):
            sample_df = pd.DataFrame({
                "ut_ms": [
                    1672531200000, 1672534800000, 1672538400000, 1672542000000,
                    1672545600000, 1672549200000, 1672552800000, 1672556400000,
                    1672560000000, 1672563600000
                ],
                "event_description": [
                    "ftl: Subsystem fault detected",
                    "emop: Emergency operation initiated",
                    "evtf: Routine event logged",
                    "saaf: Safety alert triggered",
                    "dmop: DMOP command executed",
                    "ltdata: Telemetry data recorded",
                    "solar flare: High solar activity detected",
                    "communication fault: Signal lost briefly",
                    "ftl: Another subsystem fault",
                    "evtf: Periodic event log"
                ]
            })
            csv_sample = sample_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Sample Events CSV", csv_sample, "sample_events.csv", "text/csv")
        
        if uploaded_anom is not None:
            df_anom = load_power_data(uploaded_anom)
            st.write("Data Preview (Power Data):")
            st.dataframe(df_anom.head())
            st.write("Columns:", df_anom.columns.tolist())
            st.write("Missing Values:", df_anom.isna().sum())
            if st.checkbox("Fill missing values (NPWD2372)?", key="fill_anom"):
                df_anom = fill_gaps(df_anom, "NPWD2372")
                st.write("After filling:", df_anom.isna().sum())
            st.write("Memory usage: " + get_memory_usage())
            
            st.subheader("ARIMA Parameters & Threshold")
            p = st.number_input("ARIMA p", min_value=0, max_value=10, value=2, step=1)
            d = st.number_input("ARIMA d", min_value=0, max_value=5, value=1, step=1)
            q = st.number_input("ARIMA q", min_value=0, max_value=10, value=2, step=1)
            order = (p, d, q)
            thresh_mult = st.slider("Threshold Multiplier", min_value=0.1, max_value=1.0, value=0.2, step=0.05,
                                      help="For power values between 0 and 1, use a lower multiplier")
            
            if st.button("Detect Anomalies", key="run_anom"):
                anom_results = detect_anomalies_arima(df_anom, "NPWD2372", order, thresh_mult)
                if not anom_results.empty:
                    anom_results = compute_severity(anom_results)
                    anom_detected = anom_results[anom_results["anomaly"] == 1]
                    st.write(f"Detected {anom_detected.shape[0]} anomalies.")
                    
                    fig2, ax2 = plt.subplots(figsize=(10,5))
                    ax2.plot(anom_results.index, anom_results["actual"], label="Actual")
                    anomaly_idx = anom_results.index[anom_results["anomaly"] == 1]
                    ax2.scatter(anomaly_idx, anom_results.loc[anomaly_idx, "actual"], color="red", label="Anomaly")
                    ax2.set_title("Anomaly Detection (ARIMA Residuals)")
                    ax2.legend()
                    st.pyplot(fig2)
                    
                    if uploaded_events is not None:
                        with st.spinner("Loading event data..."):
                            df_events = load_context_events(uploaded_events)
                        st.write("Events Data Preview:")
                        st.dataframe(df_events.head())
                        anom_results = fast_merge_anomalies_events(anom_results, df_events, rounding="1H")
                    else:
                        anom_results["related_event"] = "No Event"
                        anom_results["event_category"] = "unexplained"
                    
                    st.write("Anomaly Results (first 20 rows):")
                    st.dataframe(anom_results[anom_results["anomaly"] == 1].head(20))
                    
                    csv_out = anom_results.to_csv().encode("utf-8")
                    st.download_button("Download Anomaly CSV", csv_out, "anomalies.csv", "text/csv")
                else:
                    st.error("Anomaly detection failed. Check data or parameter settings.")
            st.write("Memory usage: " + get_memory_usage())

if __name__ == "__main__":
    main()
