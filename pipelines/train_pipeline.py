
import pandas as pd
from src.data.extract_timeseries_from_blocks import extract_from_hhblock
from src.features.prepare_sequences import prepare_sequences, normalize_series
from src.models.lstm_model import build_lstm
from src.models.autoencoder_model import build_autoencoder

from sklearn.model_selection import train_test_split
import numpy as np
import os

extract_from_hhblock()
df = pd.read_csv("data/processed/merged_clean.csv", parse_dates=["datetime"])
client_id = df["client_id"].unique()[0]
df_client = df[df["client_id"] == client_id].sort_values("datetime")

values_scaled, scaler = normalize_series(df_client["value"].values)
window_size = 48
X_seq = prepare_sequences(values_scaled, window_size)
X_seq = np.expand_dims(X_seq, axis=-1)

X_train, X_val = train_test_split(X_seq[:-1], test_size=0.2, shuffle=False)
y_train = X_seq[1:, -1][:len(X_train)]
y_val = X_seq[1:, -1][-len(X_val):]

lstm = build_lstm((window_size, 1))
lstm.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
lstm.save("models/lstm_model.h5")



ae = build_autoencoder((window_size, 1))
ae.fit(X_train, X_train, epochs=10, validation_data=(X_val, X_val))
ae.save("models/autoencoder_model.h5")

print("✅ Entraînement terminé et modèles sauvegardés.")
