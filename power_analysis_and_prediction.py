#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Power Analysis and Prediction Script (Final Enhanced Version)
==============================================================
This script performs data ingestion (using Pandas), preprocessing, EDA, enhanced
feature engineering (including correlation-based feature selection and additional 
polynomial interactions), regression modeling with hyperparameter tuning for 
both XGBoost and Random Forest, and time-series forecasting using classical ARIMA 
and LSTM models via the Darts library. Forecasts from ARIMA and LSTM are ensembled 
using weighted averages based on model performance. Enhanced interactive visualizations 
are produced using Plotly for better exploration and presentation.

Author: Rajat Pundir & Blessy Aaron
Date:   Updated: 17 January 2025

Dependencies:
  - pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
  - xgboost, hyperopt
  - "u8darts[all]", plotly
  - (Optional) PyTorch (for GPU usage for LSTM forecasting)

To install required packages:
    pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost hyperopt "u8darts[all]" plotly

---------------------------------------------------------------------------
Enhancements:
---------------------------------------------------------------------------
A. Feature Engineering & Data Exploration:
   - Correlation-based feature selection: Automatically drops features with near-zero 
     correlation with the target ('NPWD2372').
   - Additional polynomial features (squared and interaction terms) are added.
   
B. Regression Modeling & Hyperparameter Tuning:
   - Regression models include Linear Regression, Random Forest, and XGBoost.
   - Hyperparameter tuning is performed for both XGBoost and Random Forest using Hyperopt.
   
C. Forecasting Enhancements with Darts:
   - ARIMA and LSTM forecasts are computed after hyperparameter tuning (via Hyperopt).
   - Their forecasts are ensembled using weighted averages (inverse RMSE as weights).
   - Enhanced interactive and static visualizations (using Plotly and Matplotlib) are produced.
   
D. GPU Acceleration for LSTM:
   - The code automatically checks for PyTorch GPU availability and sets the device accordingly.
---------------------------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
import uuid
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Darts forecasting imports
from darts import TimeSeries
from darts.models import ARIMA, RNNModel
from darts.metrics import rmse

# Plotly for interactive visualizations
import plotly.graph_objs as go
import plotly.io as pio

# Check for PyTorch GPU (if using LSTM)
try:
    import torch
except ImportError:
    torch = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    filename='power_analysis_and_prediction.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --------------------
# Utility Functions (Data ingestion, processing, etc.)
# --------------------
def get_timestamp():
    """Returns the current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_uuid():
    """Generates a unique UUID."""
    return uuid.uuid4()

def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def load_data(file_paths):
    """
    Loads multiple CSV files using Pandas and concatenates them into one DataFrame.
    """
    combined = pd.DataFrame()
    for fp in file_paths:
        print(f"Reading file: {fp}")
        logging.info(f"Reading file: {fp}")
        try:
            df = pd.read_csv(fp)
            combined = pd.concat([combined, df], ignore_index=True)
        except Exception as e:
            logging.error(f"Error reading {fp}: {e}")
            raise
    logging.info(f"Combined DataFrame shape: {combined.shape}")
    return combined

def convert_timestamp(df, time_col='ut_ms'):
    """Converts numeric timestamp to datetime and sets as index."""
    df[time_col] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
    initial_shape = df.shape
    df.dropna(subset=[time_col], inplace=True)
    dropped_rows = initial_shape[0] - df.shape[0]
    logging.info(f"Dropped {dropped_rows} rows due to invalid timestamps.")
    df.set_index(time_col, inplace=True)
    return df

def basic_data_checks(df):
    """Prints basic information about the DataFrame."""
    print("\n--- Basic Data Checks ---")
    print("DataFrame shape:", df.shape)
    print("\nHead of DataFrame:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDescriptive statistics:")
    print(df.describe())
    logging.info("Performed basic data checks.")

def handle_duplicates(df):
    """Drops duplicate rows."""
    dup_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {dup_count}")
    logging.info(f"Duplicate rows before removal: {dup_count}")
    df.drop_duplicates(inplace=True)
    return df

def handle_missing_values(df):
    """Fills missing numeric values with column mean and one-hot encodes categorical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fill_dict = {col: df[col].mean() for col in numeric_cols}
    df.fillna(fill_dict, inplace=True)
    logging.info("Filled missing numeric values with column means.")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logging.info(f"One-hot encoded categorical columns: {list(categorical_cols)}")
    return df

def save_model(model, filepath):
    """
    Saves the given model to the specified file path using joblib.
    """
    try:
        joblib.dump(model, filepath)
        logging.info(f"Model saved to {filepath}")
        print(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model to {filepath}: {e}")
        raise

# --------------------
# Enhanced Feature Engineering
# --------------------
def select_features_by_correlation(df, target, threshold=0.05):
    """
    Computes correlations between each feature and the target.
    Drops features with absolute correlation below the threshold.
    """
    corr_matrix = df.corr()
    correlations = corr_matrix[target].drop(target)
    selected_features = correlations[correlations.abs() >= threshold].index.tolist()
    dropped_features = correlations[correlations.abs() < threshold].index.tolist()
    logging.info(f"Selected features (|corr| >= {threshold}): {selected_features}")
    logging.info(f"Dropped low-correlation features: {dropped_features}")
    print(f"\nSelected features (|corr| >= {threshold}): {selected_features}")
    print(f"Dropped low-correlation features: {dropped_features}")
    return selected_features

def add_polynomial_features(df, features, degree=2):
    """
    Adds squared terms and pairwise interaction terms for the provided features.
    """
    df_poly = df.copy()
    for feat in features:
        df_poly[f"{feat}_sq"] = df_poly[feat] ** 2
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            feat1 = features[i]
            feat2 = features[j]
            df_poly[f"{feat1}_x_{feat2}"] = df_poly[feat1] * df_poly[feat2]
    logging.info("Added polynomial features (squared and interaction terms).")
    print("Added polynomial features.")
    return df_poly

# --------------------
# Standard Feature Engineering (Time-based and basic interactions)
# --------------------
def feature_engineering(df):
    """
    Performs basic time-related features and interaction features.
    """
    print("\n--- Performing Feature Engineering ---")
    logging.info("Starting basic feature engineering.")
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['NPWD2372_roll3h'] = df['NPWD2372'].rolling(window=3).mean().shift(1)
    df['NPWD2401_roll3h'] = df['NPWD2401'].rolling(window=3).std().shift(1)
    df['NPWD2372_lag1h'] = df['NPWD2372'].shift(1)
    df['NPWD2372_NPWD2401_interaction'] = df['NPWD2372_lag1h'] * df['NPWD2401']
    initial_shape = df.shape
    df.dropna(inplace=True)
    dropped_rows = initial_shape[0] - df.shape[0]
    print(f"Dropped {dropped_rows} rows due to NaN values from feature engineering.")
    logging.info(f"Dropped {dropped_rows} rows from feature engineering.")
    return df

def cap_outliers_iqr(df, factor=1.5):
    """Caps outliers for numeric columns using the IQR method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        logging.info(f"Capped {col} to [{lower_bound}, {upper_bound}]")
    return df

def separate_dataframes(df):
    """Returns two copies: one raw and one with outliers capped."""
    df_anomaly = df.copy(deep=True)
    df_forecasting = cap_outliers_iqr(df.copy(deep=True), factor=1.5)
    logging.info("Separated data into anomaly and forecasting sets.")
    return df_anomaly, df_forecasting

def do_basic_eda_plots(df, run_dir, sample_cols=None):
    """Generates basic histograms and boxplots for selected numeric columns."""
    if sample_cols is None:
        sample_cols = df.select_dtypes(include=[np.number]).columns[:3].tolist()
    print(f"\n--- Basic EDA Plots for columns: {sample_cols} ---")
    logging.info(f"Generating EDA plots for {sample_cols}")
    for col in sample_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], bins=50, kde=True, color='skyblue')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"histogram_{col}.png"))
        plt.close()
    plt.figure(figsize=(16, 6))
    for i, col in enumerate(sample_cols, 1):
        plt.subplot(1, len(sample_cols), i)
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f"Boxplot: {col}")
        plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "boxplot_selected_columns.png"))
    plt.close()
    logging.info("Saved EDA plots.")

def save_cleaned_data(df, filepath):
    """Saves a Pandas DataFrame to CSV."""
    ensure_directory_exists(filepath)
    df.to_csv(filepath, index=True)
    print(f"Saved cleaned data to '{filepath}'")
    logging.info(f"Saved cleaned data to '{filepath}'")

# --------------------
# Regression Model Functions
# --------------------
def train_linear_regression(X_train, y_train):
    try:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        logging.info("Linear Regression model trained.")
        return lr
    except Exception as e:
        logging.error(f"Error training Linear Regression model: {e}")
        raise

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2):
    try:
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=42,
                                   n_jobs=-1)
        rf.fit(X_train, y_train)
        logging.info("Random Forest model trained.")
        return rf
    except Exception as e:
        logging.error(f"Error training Random Forest model: {e}")
        raise

def train_xgboost(X_train, y_train, params=None):
    try:
        if params is None:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1
            }
        xgb = XGBRegressor(**params)
        xgb.fit(X_train, y_train)
        logging.info("XGBoost model trained.")
        return xgb
    except Exception as e:
        logging.error(f"Error training XGBoost model: {e}")
        raise

def evaluate_model(model, X_test, y_test, model_name):
    try:
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        logging.info(f"{model_name} Evaluation Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        print(f"{model_name} Evaluation Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}\n")
        return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {e}")
        raise

def plot_feature_importances(model, X, run_dir, model_name='Model'):
    try:
        plt.figure(figsize=(12, 8))
        if hasattr(model, 'feature_importances_'):
            feat_imps = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            sns.barplot(x=feat_imps, y=feat_imps.index, palette='viridis')
            plt.title(f'{model_name} Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f'{model_name}_feature_importances.png'))
            plt.close()
        elif isinstance(model, LinearRegression):
            coefs = pd.Series(model.coef_, index=X.columns).abs().sort_values(ascending=False)
            sns.barplot(x=coefs, y=coefs.index, palette='magma')
            plt.title(f'{model_name} Coefficients (Absolute Values)')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f'{model_name}_coefficients.png'))
            plt.close()
        logging.info(f"Plotted feature importances for {model_name}.")
    except Exception as e:
        logging.error(f"Error plotting feature importances for {model_name}: {e}")
        raise

def save_evaluation_metrics(metrics, filepath='plots/model_evaluation_metrics.csv'):
    try:
        df_metrics = pd.DataFrame(metrics)
        ensure_directory_exists(filepath)
        df_metrics.to_csv(filepath, index=False)
        logging.info(f"Saved evaluation metrics to '{filepath}'")
    except Exception as e:
        logging.error(f"Error saving evaluation metrics: {e}")
        raise

# --------------------
# Hyperparameter Tuning Functions
# --------------------
def perform_hyperparameter_tuning_xgb(X_train, y_train, X_test, y_test):
    """Tunes XGBoost using Hyperopt."""
    try:
        def objective(space):
            params = {
                'n_estimators': int(space['n_estimators']),
                'learning_rate': space['learning_rate'],
                'max_depth': int(space['max_depth']),
                'subsample': space['subsample'],
                'colsample_bytree': space['colsample_bytree'],
                'reg_alpha': space['reg_alpha'],
                'reg_lambda': space['reg_lambda'],
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1
            }
            model = XGBRegressor(**params)
            cv = cross_val_score(model, X_train, y_train, cv=5,
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
            rmse_val = -cv.mean()
            return {'loss': rmse_val, 'status': STATUS_OK}
        
        space = {
            'n_estimators': hp.quniform('n_estimators', 100, 600, 50),
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
            'max_depth': hp.quniform('max_depth', 3, 12, 1),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': hp.uniform('reg_alpha', 0, 2.0),
            'reg_lambda': hp.uniform('reg_lambda', 0, 2.0)
        }
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=100, trials=trials, rstate=np.random.default_rng(42))
        best_params = {
            'n_estimators': int(best['n_estimators']),
            'learning_rate': best['learning_rate'],
            'max_depth': int(best['max_depth']),
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'reg_alpha': best['reg_alpha'],
            'reg_lambda': best['reg_lambda'],
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        print(f"Best XGBoost parameters: {best_params}")
        logging.info(f"Best XGBoost parameters: {best_params}")
        best_model = XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        return best_model, best_params
    except Exception as e:
        logging.error(f"Error during XGBoost hyperparameter tuning: {e}")
        raise

def perform_hyperparameter_tuning_rf(X_train, y_train, X_test, y_test):
    """Tunes Random Forest using Hyperopt."""
    try:
        def objective(space):
            params = {
                'n_estimators': int(space['n_estimators']),
                'max_depth': int(space['max_depth']),
                'min_samples_split': int(space['min_samples_split']),
                'min_samples_leaf': int(space['min_samples_leaf']),
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
            cv = cross_val_score(model, X_train, y_train, cv=5,
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
            rmse_val = -cv.mean()
            return {'loss': rmse_val, 'status': STATUS_OK}
        
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
            'max_depth': hp.quniform('max_depth', 3, 20, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1)
        }
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=50, trials=trials, rstate=np.random.default_rng(42))
        best_params = {
            'n_estimators': int(best['n_estimators']),
            'max_depth': int(best['max_depth']),
            'min_samples_split': int(best['min_samples_split']),
            'min_samples_leaf': int(best['min_samples_leaf']),
            'random_state': 42,
            'n_jobs': -1
        }
        print(f"Best Random Forest parameters: {best_params}")
        logging.info(f"Best Random Forest parameters: {best_params}")
        best_model = RandomForestRegressor(**best_params)
        best_model.fit(X_train, y_train)
        return best_model, best_params
    except Exception as e:
        logging.error(f"Error during Random Forest hyperparameter tuning: {e}")
        raise

# --------------------
# Forecasting Functions Using Darts (ARIMA & LSTM)
# --------------------
def prepare_time_series(df, target='NPWD2372'):
    """
    Converts a Pandas DataFrame (with datetime index and target column) to a Darts TimeSeries.
    """
    try:
        ts = TimeSeries.from_dataframe(df.reset_index(), 'ut_ms', target)
        logging.info("Converted DataFrame to Darts TimeSeries.")
        return ts
    except Exception as e:
        logging.error(f"Error preparing TimeSeries: {e}")
        raise

def forecast_arima(series: TimeSeries, forecast_horizon: int, order=(2, 1, 2)):
    """
    Fits an ARIMA model on the provided TimeSeries and forecasts.
    """
    try:
        model_arima = ARIMA(order=order)
        model_arima.fit(series)
        forecast = model_arima.predict(forecast_horizon)
        logging.info(f"ARIMA forecast completed with order: {order}")
        return forecast
    except Exception as e:
        logging.error(f"Error in ARIMA forecasting: {e}")
        raise

def forecast_lstm(series: TimeSeries, forecast_horizon: int,
                  input_chunk_length=24, output_chunk_length=7,
                  n_epochs=100, hidden_dim=25, num_rnn_layers=1,
                  dropout=0.1, batch_size=32, model_name="LSTM", device='cpu'):
    """
    Fits an LSTM model (via Darts' RNNModel) on the provided TimeSeries and forecasts.
    GPU acceleration is used if available.
    """
    try:
        model_lstm = RNNModel(
            model=model_name,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            hidden_dim=hidden_dim,
            num_rnn_layers=num_rnn_layers,
            dropout=dropout,
            batch_size=batch_size,
            random_state=42,
            save_checkpoints=True,
            force_reset=True,
            device=device
        )
        model_lstm.fit(series)
        forecast = model_lstm.predict(forecast_horizon)
        logging.info("LSTM forecast completed.")
        return forecast
    except Exception as e:
        logging.error(f"Error in LSTM forecasting: {e}")
        raise

def tune_arima(series: TimeSeries, forecast_horizon: int):
    """
    Uses Hyperopt to tune ARIMA order (p, d, q) parameters by minimizing RMSE.
    """
    try:
        def arima_objective(space):
            order = (int(space['p']), int(space['d']), int(space['q']))
            try:
                model = ARIMA(order=order)
                model.fit(series)
                forecast = model.predict(forecast_horizon)
                val_series = series.slice_intersect(forecast)
                error = rmse(val_series, forecast)
                return {'loss': error, 'status': STATUS_OK}
            except Exception:
                return {'loss': 1e6, 'status': STATUS_OK}
        space = {
            'p': hp.quniform('p', 0, 5, 1),
            'd': hp.quniform('d', 0, 2, 1),
            'q': hp.quniform('q', 0, 5, 1)
        }
        trials = Trials()
        best = fmin(fn=arima_objective, space=space, algo=tpe.suggest,
                    max_evals=30, trials=trials, rstate=np.random.default_rng(42))
        best_order = (int(best['p']), int(best['d']), int(best['q']))
        logging.info(f"Best ARIMA order: {best_order}")
        print(f"Best ARIMA order: {best_order}")
        return best_order
    except Exception as e:
        logging.error(f"Error tuning ARIMA: {e}")
        raise

def tune_lstm(series: TimeSeries, forecast_horizon: int, device='cpu'):
    """
    Uses Hyperopt to tune LSTM hyperparameters by minimizing RMSE on a validation split.
    """
    try:
        def lstm_objective(space):
            try:
                model = RNNModel(
                    model="LSTM",
                    input_chunk_length=int(space['input_chunk_length']),
                    output_chunk_length=int(space['output_chunk_length']),
                    n_epochs=int(space['n_epochs']),
                    hidden_dim=int(space['hidden_dim']),
                    num_rnn_layers=int(space['num_rnn_layers']),
                    dropout=space['dropout'],
                    batch_size=int(space['batch_size']),
                    random_state=42,
                    verbose=0,
                    force_reset=True,
                    device=device
                )
                train_series = series[:-forecast_horizon]
                val_series = series[-forecast_horizon:]
                model.fit(train_series)
                forecast = model.predict(forecast_horizon)
                error = rmse(val_series, forecast)
                return {'loss': error, 'status': STATUS_OK}
            except Exception:
                return {'loss': 1e6, 'status': STATUS_OK}
        space = {
            'input_chunk_length': hp.quniform('input_chunk_length', 12, 48, 12),
            'output_chunk_length': hp.quniform('output_chunk_length', 3, 14, 1),
            'n_epochs': hp.quniform('n_epochs', 50, 150, 10),
            'hidden_dim': hp.quniform('hidden_dim', 10, 50, 5),
            'num_rnn_layers': hp.quniform('num_rnn_layers', 1, 3, 1),
            'dropout': hp.uniform('dropout', 0.0, 0.3),
            'batch_size': hp.quniform('batch_size', 16, 64, 16)
        }
        trials = Trials()
        best = fmin(fn=lstm_objective, space=space, algo=tpe.suggest,
                    max_evals=30, trials=trials, rstate=np.random.default_rng(42))
        best_params = {
            'input_chunk_length': int(best['input_chunk_length']),
            'output_chunk_length': int(best['output_chunk_length']),
            'n_epochs': int(best['n_epochs']),
            'hidden_dim': int(best['hidden_dim']),
            'num_rnn_layers': int(best['num_rnn_layers']),
            'dropout': best['dropout'],
            'batch_size': int(best['batch_size'])
        }
        logging.info(f"Best LSTM parameters: {best_params}")
        print(f"Best LSTM parameters: {best_params}")
        return best_params
    except Exception as e:
        logging.error(f"Error tuning LSTM: {e}")
        raise

def ensemble_forecasts(forecasts, weights=None):
    """
    Computes a weighted ensemble forecast from a dictionary of TimeSeries forecasts.
    If weights is None, uses simple average.
    """
    try:
        forecast_vals = np.stack([f.values() for f in forecasts.values()], axis=0)
        if weights is None:
            ensemble_vals = np.mean(forecast_vals, axis=0)
        else:
            total_weight = sum(weights.get(name, 1.0) for name in forecasts.keys())
            ensemble_vals = sum(forecasts[name].values() * weights.get(name, 1.0) for name in forecasts.keys()) / total_weight
        first_forecast = next(iter(forecasts.values()))
        ensemble_forecast = first_forecast.with_values(ensemble_vals)
        logging.info("Computed ensemble forecast using weighted average.")
        return ensemble_forecast
    except Exception as e:
        logging.error(f"Error in ensembling forecasts: {e}")
        raise

# --------------------
# Enhanced Visualization Functions
# --------------------
def plot_forecast_comparison(actual: TimeSeries, forecasts_dict: dict, run_dir):
    """
    Produces interactive Plotly and static matplotlib plots for forecast comparisons.
    """
    try:
        # Interactive Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual.time_index(), y=actual.values().squeeze(),
                                 mode='lines', name='Actual'))
        for key, forecast in forecasts_dict.items():
            fig.add_trace(go.Scatter(x=forecast.time_index(), y=forecast.values().squeeze(),
                                     mode='lines', name=f'{key} Forecast', line=dict(dash='dash')))
        fig.update_layout(title='Interactive Forecast Comparison',
                          xaxis_title='Time',
                          yaxis_title='Power Consumption',
                          template='plotly_white')
        interactive_path = os.path.join(run_dir, 'forecast_comparison_interactive.html')
        fig.write_html(interactive_path)
        logging.info(f"Saved interactive forecast comparison plot to {interactive_path}")
        print(f"Interactive forecast plot saved to {interactive_path}")
        
        # Static comparison plot
        plt.figure(figsize=(14,7))
        plt.plot(actual.time_index(), actual.values(), label='Actual', color='blue', linewidth=2)
        for key, forecast in forecasts_dict.items():
            plt.plot(forecast.time_index(), forecast.values(), label=f'{key} Forecast', linestyle='--', linewidth=2)
        plt.title('Forecast Comparison')
        plt.xlabel('Time')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.tight_layout()
        full_path = os.path.join(run_dir, 'forecast_comparison_full.png')
        plt.savefig(full_path)
        plt.close()
        logging.info(f"Saved static forecast comparison plot to {full_path}")
        
        # Inset zoom plot using mpl_toolkits
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
        plt.figure(figsize=(14,7))
        plt.plot(actual.time_index(), actual.values(), label='Actual', color='blue', linewidth=2)
        for key, forecast in forecasts_dict.items():
            plt.plot(forecast.time_index(), forecast.values(), label=f'{key} Forecast', linestyle='--', linewidth=2)
        ax_main = plt.gca()
        ax_inset = zoomed_inset_axes(ax_main, zoom=3, loc='upper right')
        zoom_start = actual.time_index()[-10]
        zoom_end = actual.time_index()[-1]
        ax_inset.plot(actual.time_index(), actual.values(), label='Actual', color='blue', linewidth=2)
        for key, forecast in forecasts_dict.items():
            ax_inset.plot(forecast.time_index(), forecast.values(), label=f'{key}', linestyle='--', linewidth=2)
        ax_inset.set_xlim(zoom_start, zoom_end)
        ax_inset.set_ylim(np.min(actual.values()[-10:]) * 0.95, np.max(actual.values()[-10:]) * 1.05)
        mark_inset(ax_main, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
        plt.tight_layout()
        zoom_path = os.path.join(run_dir, 'forecast_comparison_zoom.png')
        plt.savefig(zoom_path)
        plt.close()
        logging.info(f"Saved zoomed forecast comparison plot to {zoom_path}")
        
        # Interactive violin plot for forecast errors using Plotly
        errors = {}
        for key, forecast in forecasts_dict.items():
            actual_aligned = actual.slice_intersect(forecast)
            error = np.abs(actual_aligned.values() - forecast.values()).flatten()
            errors[key] = error
        error_df = pd.DataFrame(errors)
        fig_violin = go.Figure()
        for col in error_df.columns:
            fig_violin.add_trace(go.Violin(y=error_df[col], name=col, box_visible=True, meanline_visible=True))
        fig_violin.update_layout(title="Interactive Forecast Error Distribution",
                                 yaxis_title="Absolute Error", template='plotly_white')
        violin_interactive = os.path.join(run_dir, 'forecast_error_violin_interactive.html')
        fig_violin.write_html(violin_interactive)
        logging.info(f"Saved interactive forecast error violin plot to {violin_interactive}")
        print(f"Interactive forecast error violin plot saved to {violin_interactive}")
        
        # Static violin plot
        plt.figure(figsize=(10,6))
        sns.violinplot(data=error_df)
        plt.title("Forecast Error Distribution")
        plt.ylabel("Absolute Error")
        plt.tight_layout()
        violin_static = os.path.join(run_dir, 'forecast_error_violin.png')
        plt.savefig(violin_static)
        plt.close()
        logging.info(f"Saved static forecast error violin plot to {violin_static}")
    except Exception as e:
        logging.error(f"Error in plotting forecast comparisons: {e}")
        raise

def save_summary_report(metrics, best_params_xgb, best_params_rf, filepath='plots/prediction_summary_report.txt'):
    """Saves a summary report of model metrics and best hyperparameters."""
    try:
        ensure_directory_exists(filepath)
        with open(filepath, 'w') as f:
            f.write("Model Evaluation Metrics:\n\n")
            for metric in metrics:
                f.write(f"Model: {metric['model']}\n")
                f.write(f"MAE: {metric['MAE']:.4f}\n")
                f.write(f"RMSE: {metric['RMSE']:.4f}\n")
                f.write(f"R² Score: {metric['R2']:.4f}\n\n")
            f.write(f"Best Hyperparameters for XGBoost:\n{best_params_xgb}\n\n")
            f.write(f"Best Hyperparameters for Random Forest:\n{best_params_rf}\n")
        logging.info(f"Saved summary report to '{filepath}'")
        print(f"Saved summary report to '{filepath}'")
    except Exception as e:
        logging.error(f"Error saving summary report: {e}")
        print(f"Failed to save summary report: {e}")

# --------------------
# Main Function: Regression + Forecasting
# --------------------
def main():
    try:
        # Create run directory
        timestamp = get_timestamp()
        unique_id = get_uuid()
        run_dir = os.path.join('runs', f'run_{timestamp}_{unique_id}')
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"Created run directory: {run_dir}")
        print(f"Run directory created at: {run_dir}")
        
        # Data file paths
        file_paths = [
            "/Users/rajatpundir/Desktop/Mars/train_set/power--2008-08-22_2010-07-10.csv",
            "/Users/rajatpundir/Desktop/Mars/train_set/power--2010-07-10_2012-05-27.csv",
            "/Users/rajatpundir/Desktop/Mars/train_set/power--2012-05-27_2014-04-14.csv"
        ]
        
        # Data Loading and Preprocessing using Pandas
        print("\n--- Loading Data with Pandas ---")
        df = load_data(file_paths)
        basic_data_checks(df)
        df = convert_timestamp(df, time_col='ut_ms')
        basic_data_checks(df)
        df = handle_duplicates(df)
        df = handle_missing_values(df)
        df = feature_engineering(df)
        
        # Enhanced Feature Selection via Correlation
        print("\n--- Feature Selection via Correlation ---")
        selected_features = select_features_by_correlation(df, target='NPWD2372', threshold=0.05)
        # Retain selected features along with the target
        df = df[selected_features + ['NPWD2372']]
        logging.info("Applied correlation-based feature selection.")
        
        # Add Polynomial Features
        print("\n--- Adding Polynomial Features ---")
        df = add_polynomial_features(df, features=selected_features, degree=2)
        
        # Resample Data to Hourly Frequency
        print("\n--- Resampling Data to Hourly Frequency ---")
        df_hourly = df.resample('H').mean()
        print(f"Hourly resampled DataFrame shape: {df_hourly.shape}")
        # Drop any remaining NaNs after resampling
        df_hourly = df_hourly.dropna()
        print(f"After dropping NaNs: {df_hourly.shape}")
        logging.info(f"Hourly DataFrame shape after dropping NaNs: {df_hourly.shape}")
        save_cleaned_data(df_hourly, os.path.join(run_dir, 'cleaned_power_consumption.csv'))
        
        # --------------------
        # Regression Modeling
        # --------------------
        print("\n--- Regression Modeling ---")
        target_column = 'NPWD2372'
        features = [col for col in df_hourly.columns if col != target_column]
        X = df_hourly[features]
        y = df_hourly[target_column]
        print(f"Features used for regression: {features}")
        split_point = int(len(X) * 0.8)
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        logging.info("Performed train-test split for regression.")
        print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
        
        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
        logging.info("Scaled regression features.")
        
        # Train Regression Models: Linear Regression, Random Forest, XGBoost
        lr_model = train_linear_regression(X_train_scaled, y_train)
        rf_model = train_random_forest(X_train_scaled, y_train)
        xgb_model = train_xgboost(X_train_scaled, y_train)
        
        # Evaluate Regression Models
        metrics = []
        regression_predictions = {}
        lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test, "Linear Regression")
        metrics.append(lr_metrics)
        regression_predictions['Linear Regression'] = lr_model.predict(X_test_scaled)
        plot_feature_importances(lr_model, X_train_scaled, run_dir, "Linear Regression")
        
        rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
        metrics.append(rf_metrics)
        regression_predictions['Random Forest'] = rf_model.predict(X_test_scaled)
        plot_feature_importances(rf_model, X_train_scaled, run_dir, "Random Forest")
        
        xgb_metrics = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
        metrics.append(xgb_metrics)
        regression_predictions['XGBoost'] = xgb_model.predict(X_test_scaled)
        plot_feature_importances(xgb_model, X_train_scaled, run_dir, "XGBoost")
        
        eval_metrics_path = os.path.join(run_dir, 'model_evaluation_metrics.csv')
        save_evaluation_metrics(metrics, filepath=eval_metrics_path)
        
        # Hyperparameter tuning for XGBoost
        print("\n--- Hyperparameter Tuning for XGBoost ---")
        best_xgb, best_params_xgb = perform_hyperparameter_tuning_xgb(X_train_scaled, y_train, X_test_scaled, y_test)
        tuned_xgb_metrics = evaluate_model(best_xgb, X_test_scaled, y_test, "Tuned XGBoost")
        metrics.append(tuned_xgb_metrics)
        regression_predictions['Tuned XGBoost'] = best_xgb.predict(X_test_scaled)
        plot_feature_importances(best_xgb, X_train_scaled, run_dir, "Tuned XGBoost")
        save_evaluation_metrics(metrics, filepath=eval_metrics_path)
        
        # Hyperparameter tuning for Random Forest
        print("\n--- Hyperparameter Tuning for Random Forest ---")
        best_rf, best_params_rf = perform_hyperparameter_tuning_rf(X_train_scaled, y_train, X_test_scaled, y_test)
        tuned_rf_metrics = evaluate_model(best_rf, X_test_scaled, y_test, "Tuned Random Forest")
        metrics.append(tuned_rf_metrics)
        regression_predictions['Tuned Random Forest'] = best_rf.predict(X_test_scaled)
        plot_feature_importances(best_rf, X_train_scaled, run_dir, "Tuned Random Forest")
        save_evaluation_metrics(metrics, filepath=eval_metrics_path)
        
        # Save regression models
        models_dir = os.path.join(run_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        save_model(lr_model, os.path.join(models_dir, 'linear_regression_model.pkl'))
        save_model(rf_model, os.path.join(models_dir, 'random_forest_model.pkl'))
        save_model(xgb_model, os.path.join(models_dir, 'xgboost_model.pkl'))
        save_model(best_xgb, os.path.join(models_dir, 'tuned_xgboost_model.pkl'))
        save_model(best_rf, os.path.join(models_dir, 'tuned_random_forest_model.pkl'))
        
        # --------------------
        # Forecasting using Darts: ARIMA & LSTM, then Ensemble
        # --------------------
        print("\n--- Forecasting with Darts ---")
        ts = prepare_time_series(df_hourly, target='NPWD2372')
        
        # Determine device for LSTM (GPU if available)
        device = 'cuda' if (torch and torch.cuda.is_available()) else 'cpu'
        print(f"Using {device.upper()} for LSTM forecasting.")
        logging.info(f"Using {device.upper()} for LSTM forecasting.")
        
        forecast_horizon = 30  # Forecast next 30 days
        
        # Tune and forecast ARIMA
        print("\n--- Tuning ARIMA ---")
        best_arima_order = tune_arima(ts, forecast_horizon)
        print("\n--- Forecasting with Tuned ARIMA ---")
        arima_forecast = forecast_arima(ts, forecast_horizon, order=best_arima_order)
        
        # Tune and forecast LSTM
        print("\n--- Tuning LSTM ---")
        best_lstm_params = tune_lstm(ts, forecast_horizon, device=device)
        print("\n--- Forecasting with Tuned LSTM ---")
        lstm_forecast = forecast_lstm(
            ts, forecast_horizon,
            input_chunk_length=best_lstm_params['input_chunk_length'],
            output_chunk_length=best_lstm_params['output_chunk_length'],
            n_epochs=best_lstm_params['n_epochs'],
            hidden_dim=best_lstm_params['hidden_dim'],
            num_rnn_layers=best_lstm_params['num_rnn_layers'],
            dropout=best_lstm_params['dropout'],
            batch_size=best_lstm_params['batch_size'],
            model_name="LSTM",
            device=device
        )
        
        # Compute forecasting errors
        actual_forecast = ts.slice_intersect(arima_forecast)
        arima_rmse = rmse(actual_forecast, arima_forecast)
        lstm_rmse = rmse(actual_forecast, lstm_forecast)
        
        # Weighted ensemble: inverse RMSE as weights
        weights = {
            "ARIMA": 1 / arima_rmse if arima_rmse != 0 else 1.0,
            "LSTM": 1 / lstm_rmse if lstm_rmse != 0 else 1.0
        }
        ensemble_forecast = ensemble_forecasts({"ARIMA": arima_forecast, "LSTM": lstm_forecast}, weights=weights)
        ensemble_rmse = rmse(actual_forecast, ensemble_forecast)
        print("\nForecasting Evaluation Metrics (RMSE):")
        print(f"ARIMA RMSE: {arima_rmse:.4f}")
        print(f"LSTM RMSE: {lstm_rmse:.4f}")
        print(f"Ensemble RMSE: {ensemble_rmse:.4f}\n")
        logging.info(f"Forecast RMSEs - ARIMA: {arima_rmse:.4f}, LSTM: {lstm_rmse:.4f}, Ensemble: {ensemble_rmse:.4f}")
        
        # Enhanced Interactive Visualization for Forecasting
        forecasts_dict = {"ARIMA": arima_forecast, "LSTM": lstm_forecast, "Ensemble": ensemble_forecast}
        plot_forecast_comparison(actual_forecast, forecasts_dict, run_dir)
        
        # --------------------
        # Save Summary Report
        # --------------------
        summary_report_path = os.path.join(run_dir, 'prediction_summary_report.txt')
        save_summary_report(metrics, best_params_xgb=best_xgb.get_params(), best_params_rf=best_params_rf, filepath=summary_report_path)
        
        print("\nScript completed successfully.")
        logging.info("Script completed successfully.")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred in main(): {e}")
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
