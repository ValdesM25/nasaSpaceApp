""" import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib as mtl
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

<<<<<<< HEAD
    #Interpretavilidad

    # Calcular SHAP values
    #shap_values_single = explainer.shap_values(X_scaled)

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    label = int(prob > 0.5)
=======
    # --- PREDECIR ---
    probs = model.predict_proba(X_scaled)[:, 1]       # probabilidades clase positiva
    labels = (probs > 0.5).astype(int)                # etiquetas binarias
>>>>>>> 71784c14add8b95b14010450c0660830b1022097

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

<<<<<<< HEAD
    return result
 """

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib as mtl
from ml.util.features import create_efficient_features
import json

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

    #Interpretavilidad
    # ========== BLOQUE DE EXPLICACIÓN SHAP - AGREGAR DESPUÉS DE "Interpretabilidad" ==========
    
    # Cargar explainer SHAP para el modelo
    explainer_path = f'ml/models/shap_explainers/{model_name}_explainer.pkl'
    explainer = joblib.load(explainer_path)
    
    # Calcular SHAP values para esta predicción
    shap_values = explainer.shap_values(X_scaled)
    
    # Para clasificación binaria, usar valores de la clase 1 (CONFIRMED)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 values
        expected_value = explainer.expected_value[1]
    else:
        expected_value = explainer.expected_value
    
    # Obtener contribuciones de features
    feature_names = [
        'transit_period', 'transit_duration', 'transit_depth', 'planet_radius',
        'star_temperature', 'star_radius', 'star_surface_gravity', 'star_brightness',
        'period_duration_ratio', 'snr_approximation', 'depth_discrepancy',
        'star_luminosity', 'transit_probability', 'log_transit_depth', 'log_transit_period'
    ]
    
    # Calcular top features más influyentes
    contributions = []
    for i, feature in enumerate(feature_names):
        contributions.append({
            'feature': feature,
            'value': float(X_scaled[0][i]),
            'shap_value': float(shap_values[0][i]),
            'impact': abs(float(shap_values[0][i]))
        })
    
    # Ordenar por impacto y tomar top 5
    contributions.sort(key=lambda x: x['impact'], reverse=True)
    top_features = contributions[:5]
    
    # ========== FIN DEL BLOQUE DE EXPLICACIÓN ==========

    # Calcular SHAP values
    #shap_values_single = explainer.shap_values(X_scaled)

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    label = int(prob > 0.5)

        # Results - MODIFICAR ESTE BLOQUE
    result = {
        "model": model_name,
        "probability_confirmed": float(prob),
        "prediction": "CONFIRMED" if label == 1 else "FALSE POSITIVE",
        "explanation": {
            "base_value": float(expected_value),
            "top_features": [
                {
                    'feature': feat['feature'],
                    'contribution': feat['shap_value'],
                    'absolute_impact': feat['impact'],
                    'actual_value': feat['value']
                }
                for feat in top_features
            ]
        }
    }

    return result