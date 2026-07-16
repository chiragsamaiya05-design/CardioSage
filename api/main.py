from fastapi import FastAPI
import pandas as pd

from api.schemas import HeartRequest
from api.model_loader import model

app = FastAPI(
    title="CardioSage API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "CardioSage API is running."
    }


@app.post("/predict")
def predict(data: HeartRequest):

    df = pd.DataFrame([data.model_dump()])

    probability = float(model.predict_proba(df)[0][1])

    prediction = (
        "High Risk"
        if probability >= 0.5
        else "Low Risk"
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }