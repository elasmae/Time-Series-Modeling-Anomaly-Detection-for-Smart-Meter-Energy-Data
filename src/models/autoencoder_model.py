
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(64, activation="relu", return_sequences=False)(inputs)
    repeated = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(64, activation="relu", return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(1))(decoded)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mae")
    return model
