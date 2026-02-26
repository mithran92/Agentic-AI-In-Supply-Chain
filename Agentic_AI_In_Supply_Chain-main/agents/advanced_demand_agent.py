import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "demand_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["sales", "price", "holiday", "promotion", "temperature", "fuel_price"]


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("sales.csv not found in data folder.")
    return pd.read_csv(DATA_PATH)


# -----------------------------
# Prepare Time Series Data
# -----------------------------
def prepare_data(df):

    window = 5

    if len(df) <= window:
        raise ValueError("Not enough data to create time windows.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])

    X, y = [], []

    for i in range(len(scaled) - window):
        X.append(scaled[i:i + window])
        y.append(scaled[i + window][0])  # sales is target

    return np.array(X), np.array(y), scaler


# -----------------------------
# Build LSTM Model
# -----------------------------
def build_model(input_shape):

    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    return model


# -----------------------------
# Load or Train Model
# -----------------------------
def load_or_train():

    df = load_data()
    X, y, scaler = prepare_data(df)

    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model = load_model(MODEL_PATH, compile=False)
    else:
        print("Training new model...")
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=30, verbose=0)
        model.save(MODEL_PATH)

    return model, scaler, df


# -----------------------------
# Predict Next Demand
# -----------------------------
def predict_demand_lstm():

    model, scaler, df = load_or_train()

    last_window = df[FEATURES].tail(5).values

    scaled_window = scaler.transform(last_window)
    scaled_window = scaled_window.reshape(1, 5, len(FEATURES))

    prediction_scaled = model.predict(scaled_window, verbose=0)

    # Create dummy row for inverse scaling
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, 0] = prediction_scaled[0, 0]

    prediction = scaler.inverse_transform(dummy)[0, 0]

    prediction = max(0, int(prediction))

    print("Predicted Demand:", prediction)

    return prediction