from fastapi import FastAPI, Header, HTTPException
import joblib
import os
import numpy as np

app = FastAPI(title="Battery Management ML API")

MODEL_FILE = "models/battery_thermal_model.pkl"

# Demo API key for security testing
API_KEY = os.getenv("BMS_API_KEY")

model = joblib.load(MODEL_FILE)


@app.get("/")
def home():
    return {
        "message": "Battery Management ML API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(
    temperature: float,
    voltage: float,
    current: float,
    coolant_flow: float,
    x_api_key: str | None = Header(default=None)
):
    # API key authentication
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key"
        )

    data = np.array([
        [temperature, voltage, current, coolant_flow]
    ])

    prediction = model.predict(data)[0]

    return {
        "thermal_index": round(float(prediction), 3)
    }