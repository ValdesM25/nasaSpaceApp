import joblib
import numpy as np
import pandas as pd
from ml.util.features import create_efficient_features

# Load models once
rf_model = joblib.load('ml/models/randomforest_model.pkl')
xgb_model = joblib.load('ml/models/xgboost_model.pkl')
lgbm_model = joblib.load('ml/models/lightgbm_model.pkl')
scaler = joblib.load('ml/models/scaler.pkl')

models = {
    "random_forest": rf_model,
    "xgboost": xgb_model,
    "lightgbm": lgbm_model
}

def predict_exoplanet(df, model_name):
    # Generate additional features
    df = create_efficient_features(df)
    df = df.drop(columns=['transit_depth.1', 'category'])

    # Scale data
    X_scaled = scaler.transform(df)


    # Verify that model exists
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(models.keys())}")

    # Obtain model
    model = models[model_name]

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    label = int(prob > 0.5)

    # Results
    result = {
        "model": model_name,
        "probability_confirmed": float(prob),
        "prediction": "CONFIRMED" if label == 1 else "FALSE POSITIVE"
    }

    return result
