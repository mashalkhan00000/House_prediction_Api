import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Create dataset
np.random.seed(42)
n = 500
data = pd.DataFrame({
    'bedrooms': np.random.randint(1, 6, n),
    'bathrooms': np.random.randint(1, 4, n),
    'sqft': np.random.randint(500, 5000, n),
    'age_years': np.random.randint(1, 50, n),
    'garage': np.random.randint(0, 3, n),
})
data['price'] = (
    data['sqft'] * 150 +
    data['bedrooms'] * 10000 +
    data['bathrooms'] * 8000 +
    data['garage'] * 15000 -
    data['age_years'] * 500 +
    np.random.randint(-20000, 20000, n)
)

X = data.drop('price', axis=1)
y = data['price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'model.pkl')
print("✅ model.pkl created successfully!")
