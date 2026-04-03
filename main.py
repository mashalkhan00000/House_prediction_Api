from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import gradio as gr

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

# ---- GRADIO UI (added below — nothing above is changed) ----
def gradio_predict(bedrooms, bathrooms, sqft, age_years, garage):
    data = [[bedrooms, bathrooms, sqft, age_years, garage]]
    prediction = model.predict(data)[0]
    return f"${round(prediction, 2):,}"

demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms"),
        gr.Number(label="Sqft"),
        gr.Number(label="Age of House (years)"),
        gr.Number(label="Garage (0 = No, 1 = Yes)"),
    ],
    outputs=gr.Text(label="Predicted Price"),
    title="House Price Predictor — Somaan Khan",
    description="Fill in the house details and get an instant price prediction!"
)

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/demo")
