# Case-study Parametry.AI

### 📘 Project Title: Satellite Anomaly Detection and Power Consumption Forecasting Using Machine Learning  
🎓 Master's in Applied Data Science and Analytics  
🏫 SRH Hochschule Heidelberg, Germany

---

## 📌 Overview

This project presents a comprehensive machine learning pipeline for forecasting power consumption and detecting anomalies in telemetry data from the **Mars Express Orbiter (MEX)**. The study leverages **SARIMA**, **Isolation Forest**, and **Rolling Z-Score** models on multi-source telemetry data provided by **ESA** (European Space Agency).

---

## 🎯 Objectives

- Accurately **forecast power consumption** at daily and monthly intervals.
- Detect **operational anomalies** using statistical and machine learning methods.
- Perform extensive **feature engineering** from raw telemetry datasets.
- Provide actionable insights for **proactive mission control** and spacecraft reliability.

---

## 🧠 Methods Used

### 🔧 Power Consumption Forecasting:
- **Models:** SARIMA, Linear Regression, Random Forest, XGBoost
- **Best Performer:** SARIMA  
  - Daily RMSE: `0.0375`, MAPE: `3.56%`
  - Monthly RMSE: `0.0448`, MAPE: `4.22%`
- **Preprocessing:** Stationarity testing, time aggregation, outlier handling

### 🚨 Anomaly Detection:
- **Methods:** Rolling Z-Score, Isolation Forest (chosen model)
- **Results:** Isolation Forest detected `310` anomalies, outperforming statistical methods
- **Contamination Rate:** 1%

---

## 📂 Dataset & Sources

- Telemetry Data from ESA’s **Mars Express Orbiter** (2008–2014)
- Raw Data Files:  
  - DMOP (Detailed Mission Operations Plan)  
  - EVTF (Event Timeline)  
  - FTL (Flight Dynamics Timeline)  
  - SAAF (Solar Aspect Angles)  
  - LTDATA (Long-Term Distance & Positioning)  
  - Power Lines (NPWD2372)

---

## 🏗️ Feature Engineering

- **Solar Energy Intake** using cosine-transformed solar angles and Sun-Mars distance
- **Subsystem Interactions** based on command co-occurrences
- **Event Flags** like LOS (Loss of Signal), AOS (Acquisition of Signal)
- **Pointing Events** (NADIR, EARTH, SLEW) via one-hot encoding

---

## 📈 Results Summary

| Task                     | Best Model     | Metric   | Value       |
|--------------------------|----------------|----------|-------------|
| Daily Power Forecasting  | SARIMA         | RMSE     | 0.0375      |
| Monthly Forecasting      | SARIMA         | MAPE     | 4.22%       |
| Anomaly Detection        | IsolationForest| Anomalies| 310         |

---

## 🧩 Challenges

- Merging multi-resolution telemetry sources
- Complex feature extraction and domain transformations
- Sensitivity balancing in anomaly detection
- High computational
