
#  Time Series Modeling & Anomaly Detection

This project is an **end-to-end time series modeling pipeline** including:
- Statistical Models: ARIMA, SARIMA, Holt-Winters, Prophet
- Deep Learning: LSTM, AutoEncoder
- Anomaly Detection: AutoEncoder, One-Class SVM, Isolation Forest



##  Structure

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

---

##  Quickstart

```bash
# Build container
make docker-build

# Run in Jupyter
make docker-run

# Or local training
make train
```


