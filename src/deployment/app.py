from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
app = FastAPI(title="Water Leak Detector")

#old method
# model = joblib.load("models/lstm_model.pkl")
# scaler = joblib.load("models/scaler.pkl")


model = load_model("models/lstm_model.h5", custom_objects={"AttentionLayer": AttentionLayer})
# You MUST pass your custom layer to load_model
scaler = joblib.load("models/feature_scaler.pkl") # Load the correct scaler
target_scaler = joblib.load("models/target_scaler.pkl")

class InputData(BaseModel):
    flow_history: list

@app.post("/predict")
def predict(data: InputData):
    seq = np.array(data.flow_history[-168:]).reshape(1, 168, 1)
    seq_scaled = scaler.transform(seq.reshape(-1, 1)).reshape(1, 168, 1)
    pred = model.predict(seq_scaled)[0][0]
    actual = seq_scaled[0][-1][0]
    anomaly = abs(pred - actual) > 2.5
    return {"predicted_flow": float(pred), "leak_alert": anomaly}