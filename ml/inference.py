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
    # Generar características adicionales
    df = create_efficient_features(df)

    # Eliminar columnas no deseadas si existen
    for col in ['transit_depth.1', 'category']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Escalar datos
    X_scaled = scaler.transform(df)

    # Verificar modelo
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(models.keys())}")

    # Obtener modelo
    model = models[model_name]

    # --- PREDECIR ---
    probs = model.predict_proba(X_scaled)[:, 1]       # probabilidades clase positiva
    labels = (probs > 0.5).astype(int)                # etiquetas binarias

    # Si solo hay una fila → devolver resultado individual
    if len(df) == 1:
        return {
            "model": model_name,
            "probability_confirmed": float(probs[0]),
            "prediction": "CONFIRMED" if labels[0] == 1 else "FALSE POSITIVE"
        }

    # Si hay múltiples filas → devolver lista de resultados
    results = []
    for i, (p, l) in enumerate(zip(probs, labels)):
        results.append({
            "index": int(i),
            "probability_confirmed": float(p),
            "prediction": "CONFIRMED" if l == 1 else "FALSE POSITIVE"
        })

    return {
        "model": model_name,
        "results": results
    }

