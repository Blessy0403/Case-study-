#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly Detection & Classification for Mars Express Power Consumption
------------------------------------------------------------------------------
This script:
  1. Loads raw power consumption CSVs from the train_set.
  2. Uses a statsmodels ARIMA (order=(2,1,2)) to compute residuals and flag anomalies.
  3. Loads context/event CSVs (dmop, evtf, ftl, ltdata, saaf) from train_set.
     - It checks for a timestamp in "ut_ms" or "utb_ms" (e.g., in ftl files) or any column with "time".
  4. Merges event data with anomalies (using a ±1 hour window) to pre-label anomalies as:
       - Operational (if event mentions commands/faults)
       - Natural Effect (if event mentions CME/solar flare)
       - Other (if no event found or not matching)
  5. Computes anomaly severity scores ("Low"/"Medium"/"High") based on the residual magnitude.
  6. If there’s enough category diversity, trains Random Forest and Gradient Boosting classifiers;
     generates a SHAP summary for the Random Forest.
  7. Saves the final anomaly table to CSV.
  
Ensure the file paths below match your environment.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
import warnings

# Attempt to import SHAP; if unavailable, we will skip SHAP-related steps.
try:
    import shap
except Exception as e:
    shap = None
    logging.warning("SHAP could not be imported; skipping SHAP analysis.")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------
# CONFIGURATION & PATHS
# ------------------------
RAW_FILES = [
    "/Users/rajatpundir/Desktop/Mars/train_set/power--2008-08-22_2010-07-10.csv",
    "/Users/rajatpundir/Desktop/Mars/train_set/power--2010-07-10_2012-05-27.csv",
    "/Users/rajatpundir/Desktop/Mars/train_set/power--2012-05-27_2014-04-14.csv"
]

# Folder containing event/context CSVs (dmop, evtf, ftl, ltdata, saaf)
EVENTS_FOLDER = "/Users/rajatpundir/Desktop/Mars/train_set/"
TARGET_COL = "NPWD2372"

# ------------------------
# FUNCTIONS
# ------------------------
def load_raw_power_data(file_list):
    """Loads and concatenates raw power CSV files."""
    df_list = []
    for fpath in file_list:
        try:
            temp = pd.read_csv(fpath, engine="python")
            temp['ut_ms'] = pd.to_datetime(temp['ut_ms'], unit='ms', errors='coerce')
            df_list.append(temp)
        except Exception as e:
            logging.error(f"Error reading {fpath}: {e}")
    df = pd.concat(df_list, ignore_index=True)
    df.dropna(subset=['ut_ms'], inplace=True)
    df.sort_values('ut_ms', inplace=True)
    df.set_index('ut_ms', inplace=True)
    return df

def detect_anomalies_arima(df, target=TARGET_COL, order=(2,1,2), threshold_multiplier=3):
    """
    Fits a statsmodels ARIMA model on the target series, computes residuals,
    and flags anomalies where |residual| > (mean + threshold_multiplier * std).
    """
    series = df[target].dropna()
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    pred = model_fit.predict(start=series.index[0], end=series.index[-1])
    residuals = series - pred
    mean_res = residuals.mean()
    std_res = residuals.std()
    threshold = mean_res + threshold_multiplier * std_res
    anomaly_flags = (residuals.abs() > threshold).astype(int)
    anom_df = pd.DataFrame({
        'actual': series,
        'predicted': pred,
        'residual': residuals,
        'anomaly': anomaly_flags
    })
    return anom_df

def load_context_events(folder_path):
    """
    Loads all CSV files in folder_path that are not power files.
    Checks for "ut_ms", then "utb_ms", else any column containing "time".
    Expects an 'event_description' column if available.
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    frames = []
    for f in all_files:
        if "power--" in os.path.basename(f):
            continue
        try:
            df = pd.read_csv(f, engine="python")
            if "ut_ms" in df.columns:
                df["timestamp"] = pd.to_datetime(df["ut_ms"], unit="ms", errors="coerce")
            elif "utb_ms" in df.columns:
                df["timestamp"] = pd.to_datetime(df["utb_ms"], unit="ms", errors="coerce")
            else:
                time_cols = [col for col in df.columns if "time" in col.lower()]
                if time_cols:
                    df["timestamp"] = pd.to_datetime(df[time_cols[0]], errors="coerce")
                else:
                    logging.warning(f"No timestamp column found in {f}. Skipping file.")
                    continue
            frames.append(df)
        except Exception as e:
            logging.warning(f"Error reading {f}: {e}")
            continue
    if frames:
        events_df = pd.concat(frames, ignore_index=True)
        events_df.dropna(subset=["timestamp"], inplace=True)
        events_df.sort_values("timestamp", inplace=True)
        return events_df
    else:
        return pd.DataFrame()

def label_event_category(desc):
    """
    Labels an event description:
      - 'Natural Effect' if contains 'cme', 'solar flare', or 'natural'
      - 'Operational' if contains 'operation', 'comm', 'command', 'switch', 'fault', or 'dmop'
      - 'Other' otherwise.
    """
    desc = str(desc).lower()
    if any(x in desc for x in ['cme', 'solar flare', 'natural']):
        return 'Natural Effect'
    elif any(x in desc for x in ['operation', 'comm', 'command', 'switch', 'fault', 'dmop']):
        return 'Operational'
    else:
        return 'Other'

def merge_anomalies_with_events(anom_df, events_df, time_window="1H"):
    """
    For each anomaly timestamp in anom_df, finds any event in events_df within ± time_window.
    Assigns the event's category if found; otherwise defaults to 'Other'.
    """
    anom_df = anom_df.copy()
    anom_df["anomaly_category"] = "Other"
    for idx in anom_df.index:
        start = idx - pd.Timedelta(time_window)
        end = idx + pd.Timedelta(time_window)
        subset = events_df[(events_df["timestamp"] >= start) & (events_df["timestamp"] <= end)]
        if not subset.empty:
            # If 'event_description' exists, use it; else default to "Other"
            if "event_description" in subset.columns:
                cat = label_event_category(subset.iloc[0]["event_description"])
            else:
                cat = "Other"
            anom_df.at[idx, "anomaly_category"] = cat
    return anom_df

def compute_severity_scores(anom_df):
    """
    Computes anomaly severity based on the absolute residual.
    Uses the 33rd and 66th percentiles to assign:
       - "Low" if below 33rd percentile,
       - "Medium" if between 33rd and 66th percentile,
       - "High" if above 66th percentile.
    """
    anom_only = anom_df[anom_df["anomaly"] == 1]
    if anom_only.empty:
        anom_df["severity"] = np.nan
        return anom_df
    resid_abs = anom_only["residual"].abs()
    q33 = np.percentile(resid_abs, 33)
    q66 = np.percentile(resid_abs, 66)
    def get_severity(val):
        if val < q33:
            return "Low"
        elif val < q66:
            return "Medium"
        else:
            return "High"
    severity_series = resid_abs.apply(get_severity)
    anom_df["severity"] = "N/A"
    anom_df.loc[severity_series.index, "severity"] = severity_series
    return anom_df

def extract_classification_features(anom_df):
    """
    Extracts features for classification:
      - Absolute residual, Hour of day, and Day of week.
    """
    feats = pd.DataFrame(index=anom_df.index)
    feats["resid_abs"] = anom_df["residual"].abs()
    feats["hour"] = anom_df.index.hour
    feats["dayofweek"] = anom_df.index.dayofweek
    return feats

# ------------------------
# MAIN EXECUTION
# ------------------------
def main():
    logging.info("Loading raw power data from CSVs...")
    df_raw = load_raw_power_data(RAW_FILES)
    logging.info(f"Combined shape: {df_raw.shape}")
    
    logging.info(f"Detecting anomalies using ARIMA residuals (order=(2,1,2)) on '{TARGET_COL}'...")
    anom_df = detect_anomalies_arima(df_raw, target=TARGET_COL, order=(2,1,2), threshold_multiplier=3)
    num_anom = int(anom_df["anomaly"].sum())
    logging.info(f"Detected {num_anom} anomalies.")
    
    # Save anomaly plot for reference
    plt.figure(figsize=(12, 5))
    plt.plot(anom_df.index, anom_df["actual"], label="Actual")
    plt.scatter(anom_df.index[anom_df["anomaly"]==1],
                anom_df["actual"][anom_df["anomaly"]==1],
                color="red", label="Anomaly")
    plt.title("Anomaly Detection (ARIMA Residuals)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("anomaly_plot.png")
    plt.close()
    logging.info("Saved anomaly_plot.png")
    
    logging.info("Loading context/event data from train folder...")
    events_df = load_context_events(EVENTS_FOLDER)
    logging.info(f"Events data shape: {events_df.shape}")
    
    if events_df.empty:
        logging.warning("No event data found; assigning all anomalies as 'Other'.")
        merged_anom = anom_df.copy()
        merged_anom["anomaly_category"] = "Other"
    else:
        merged_anom = merge_anomalies_with_events(anom_df, events_df, time_window="1H")
    
    merged_anom = compute_severity_scores(merged_anom)
    
    # Only perform classification if there is sufficient category diversity
    anom_only = merged_anom[merged_anom["anomaly"] == 1].copy()
    if anom_only.empty:
        logging.info("No anomalies to classify. Saving severity results only.")
        merged_anom.to_csv("detected_anomalies_with_classification.csv")
        return
    if anom_only["anomaly_category"].nunique() < 2:
        logging.warning("Not enough category diversity to train a classifier. Saving severity results only.")
        merged_anom.to_csv("detected_anomalies_with_classification.csv")
        return

    feats = extract_classification_features(anom_only)
    labels = anom_only["anomaly_category"]
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    X_train, X_test, y_train, y_test = train_test_split(feats, labels, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    for model, name in zip([rf, gb], ["RandomForest", "GradientBoosting"]):
        preds = model.predict(X_test)
        logging.info(f"\n{name} Classification Report:\n{classification_report(y_test, preds)}")
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")
    
    # SHAP analysis for Random Forest (gracefully skip if SHAP is not available or fails)
    if shap is not None:
        try:
            explainer = shap.Explainer(rf, X_train)
            shap_values = explainer(X_train)
            shap.summary_plot(shap_values, X_train, show=False)
            plt.savefig("shap_summary_rf.png", bbox_inches="tight")
            plt.close()
            logging.info("Saved SHAP summary plot as shap_summary_rf.png")
        except Exception as e:
            logging.warning(f"SHAP analysis skipped: {e}")
    else:
        logging.warning("SHAP is not installed; skipping SHAP analysis.")
    
    anom_only["predicted_rf"] = rf.predict(feats)
    anom_only["predicted_gb"] = gb.predict(feats)
    merged_anom.update(anom_only)
    
    output_file = "detected_anomalies_with_classification_and_severity.csv"
    merged_anom.to_csv(output_file)
    logging.info(f"Saved final anomaly results to {output_file}")

if __name__ == "__main__":
    main()
