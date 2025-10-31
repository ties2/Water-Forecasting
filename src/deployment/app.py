from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Leak Detector API")

model = joblib.load("models/trained_model.pkl")
scaler = joblib.load("models/scaler.pkl")

class InputData(BaseModel):
    flow_history: list

@app.post("/predict")
def predict(data: InputData):
    seq = np.array(data.flow_history[-168:]).reshape(1, 168, 1)
    seq = scaler.transform(seq)
    pred = model.predict(seq)
    anomaly = abs(pred[0][0] - seq[0][-1][0]) > 3  # simplified
    return {"predicted_flow": float(pred[0][0]), "leak_alert": anomaly}