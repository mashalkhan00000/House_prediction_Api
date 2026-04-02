from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using Machine Learning — Built by Somaan Khan",
    version="1.0.0"
)

# Define input data structure
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: int
    sqft: int
    age_years: int
    garage: int

# Home route
@app.get("/")
def home():
    return {
        "message": "Welcome to House Price Prediction API!",
        "developer": "Somaan Khan - Data Scientist",
        "usage": "Send a POST request to /predict with house features"
    }

# Prediction route
@app.post("/predict")
def predict(features: HouseFeatures):
    data = [[
        features.bedrooms,
        features.bathrooms,
        features.sqft,
        features.age_years,
        features.garage
    ]]
    prediction = model.predict(data)[0]
    return {
        "predicted_price": f"${round(prediction, 2):,}",
        "input_features": features.dict()
    }
