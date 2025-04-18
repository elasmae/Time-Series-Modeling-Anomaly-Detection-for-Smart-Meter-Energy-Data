
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from src.features.prepare_sequences import prepare_sequences, normalize_series
from src.visualization.plot_anomalies import plot_series_with_anomalies

df = pd.read_csv("data/processed/merged_clean.csv", parse_dates=["datetime"])
client_id = df["client_id"].unique()[0]
df_client = df[df["client_id"] == client_id].sort_values("datetime")

values_scaled, scaler = normalize_series(df_client["value"].values)
window_size = 48
X_seq = prepare_sequences(values_scaled, window_size)
X_seq = np.expand_dims(X_seq, axis=-1)

# Charger modèle AutoEncoder
ae = load_model("models/autoencoder_model.h5")
X_pred = ae.predict(X_seq)
mae = np.mean(np.abs(X_pred - X_seq), axis=(1, 2))
threshold = np.percentile(mae, 95)
anomalies = mae > threshold

timestamps = df_client["datetime"].iloc[window_size:]
values = df_client["value"].iloc[window_size:]

plot_series_with_anomalies(timestamps.values, values.values, anomalies, title="Anomalies détectées")
