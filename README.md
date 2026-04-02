# 🏠 House Price Prediction API
**Built by Somaan Khan — Data Scientist**

A Machine Learning API that predicts house prices based on key features.

## 🚀 Live Demo
Once deployed on Render, your API will be live at:
`https://house-price-api.onrender.com`

## 📊 Model Performance
- Algorithm: Random Forest Regressor
- R2 Score: 99.11%

## 🔧 Features Used
| Feature | Description |
|---|---|
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| sqft | Total square footage |
| age_years | Age of the house in years |
| garage | Number of garage spaces |

## 📡 API Endpoints

### GET /
Returns welcome message and usage info.

### POST /predict
Predict house price.

**Request Body:**
```json
{
  "bedrooms": 3,
  "bathrooms": 2,
  "sqft": 1500,
  "age_years": 10,
  "garage": 1
}
```

**Response:**
```json
{
  "predicted_price": "$285,000.00",
  "input_features": {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft": 1500,
    "age_years": 10,
    "garage": 1
  }
}
```

## 🛠️ Run Locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Then open: http://localhost:8000/docs

