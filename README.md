# Smart Meter Time Series Modeling and Anomaly Detection

A machine learning pipeline for forecasting and anomaly detection on high-frequency smart meter electricity consumption data.

## Overview

Electricity consumption data is inherently temporal, noisy, and influenced by seasonal and behavioral patterns.  
This work aims to:

- Forecast short- and medium-term energy consumption
- Detect abnormal usage patterns
- Compare statistical and deep learning time-series approaches within a unified pipeline

## Dataset

The dataset contains half-hourly electricity consumption records from multiple smart meter clients.  
Raw data is transformed into a timestamped long-format time series for modeling.

## Methodology

### Statistical Forecasting
- ARIMA
- SARIMA
- Holt-Winters
- Prophet

### Deep Learning
- LSTM sequence modeling
- AutoEncoder reconstruction modeling

### Anomaly Detection
- AutoEncoder reconstruction error
- Isolation Forest
- One-Class SVM

## Evaluation

Forecasting models are evaluated using regression metrics such as MAE and RMSE.  
Anomaly detection models rely on reconstruction errors and unsupervised anomaly scoring mechanisms.

## Project Structure

```
.
├── src/                # Modular code (data, models, features, viz)
├── pipelines/          # Train / Inference pipelines
├── notebooks/          # Exploration & modeling notebooks
├── data/               # Raw and processed data
├── models/             # Trained models
├── Dockerfile          # Docker container
├── Makefile            # Automation commands
└── requirements.txt    # Python dependencies
```

## Quickstart

```bash
# Build Docker container
make docker-build

# Run Jupyter environment
make docker-run

# Train models locally
make train
```
