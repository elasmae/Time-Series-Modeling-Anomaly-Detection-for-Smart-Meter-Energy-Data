
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_sequences(values, window_size):
    sequences = []
    for i in range(len(values) - window_size):
        sequences.append(values[i:i + window_size])
    return np.array(sequences)

def normalize_series(values):
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    return values_scaled, scaler
