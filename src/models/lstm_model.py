
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation="relu", input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model
