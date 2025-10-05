import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
from util.features import create_efficient_features
import warnings
warnings.filterwarnings('ignore')

# ======================================================
# 1. Cargar datos
# ======================================================

data = pd.read_csv('datasets/final/keplern.csv')
data = data.drop('transit_depth.1', axis=1)
labeled_data = data[data['category'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

le = LabelEncoder()
labeled_data['target'] = le.fit_transform(labeled_data['category'])

labeled_data_eff = create_efficient_features(labeled_data)

features = [
    'transit_period', 'transit_duration', 'transit_depth', 'planet_radius',
    'star_temperature', 'star_radius', 'star_surface_gravity', 'star_brightness',
    'period_duration_ratio', 'snr_approximation', 'depth_discrepancy',
    'star_luminosity', 'transit_probability', 'log_transit_depth', 'log_transit_period'
]

X = labeled_data_eff[features].fillna(0)
y = labeled_data_eff['target']

# ======================================================
# 3. Split y escalado
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar scaler
joblib.dump(scaler, 'models/scaler.pkl')

print(f"Entrenamiento: {X_train_scaled.shape}")
print(f"Prueba: {X_test_scaled.shape}")

# ======================================================
# 4. Definir y entrenar modelos
# ======================================================

print("\nEntrenando modelos rápidos...")

xgb_fast = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgbm_fast = LGBMClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

rf_fast = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

models = {
    'XGBoost': xgb_fast,
    'LightGBM': lgbm_fast,
    'RandomForest': rf_fast
}

# ======================================================
# 5. Entrenamiento + evaluación + guardado
# ======================================================

results = {}
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Crear directorios necesarios
import os
os.makedirs('models/shap_explainers', exist_ok=True)
os.makedirs('models/results_log', exist_ok=True)

for name, model in models.items():
    print(f"\n🔹 Entrenando {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Guardar modelo
    model_path = f"models/{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_path)

    # ======================================================
    # CREAR Y GUARDAR EXPLAINERS DE SHAP
    # ======================================================
    print(f"🔄 Generando explainer SHAP para {name}...")
    
    try:
        # Crear explainer SHAP
        explainer = shap.TreeExplainer(model)
        
        # Preparar datos de muestra para SHAP (usar subset para eficiencia)
        sample_size = min(100, len(X_test_scaled))
        X_sample = X_test_scaled[:sample_size]
        
        # Calcular SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Guardar explainer
        explainer_path = f"models/shap_explainers/{name.lower().replace(' ', '_')}_explainer.pkl"
        joblib.dump(explainer, explainer_path)
        
        # Guardar SHAP values de muestra
        shap_values_path = f"models/shap_explainers/{name.lower().replace(' ', '_')}_shap_values.pkl"
        joblib.dump(shap_values, shap_values_path)
        
        print(f"✅ Explainers SHAP para {name} guardados correctamente")
        
    except Exception as e:
        print(f"⚠️ Error con SHAP para {name}: {str(e)}")
        # En caso de error, intentar con KernelExplainer como fallback
        try:
            print("🔄 Intentando con KernelExplainer...")
            background = shap.sample(X_train_scaled, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            
            explainer_path = f"models/shap_explainers/{name.lower().replace(' ', '_')}_kernel_explainer.pkl"
            joblib.dump(explainer, explainer_path)
            print(f"✅ KernelExplainer para {name} guardado como fallback")
            
        except Exception as e2:
            print(f"❌ Fallback también falló para {name}: {str(e2)}")

    # Guardar resultado
    results[name] = {
        "accuracy": round(acc, 4),
        "accuracy_percent": round(acc * 100, 2),
        "model_path": model_path,
        "shap_explainer_path": f"models/shap_explainers/{name.lower().replace(' ', '_')}_explainer.pkl",
        "train_size": len(y_train),
        "test_size": len(y_test),
        "timestamp": timestamp
    }

    print(f"✅ {name} guardado ({acc*100:.2f}% accuracy)")

# ======================================================
# 6. Guardar metadata de features para SHAP
# ======================================================

feature_metadata = {
    'feature_names': features,
    'class_names': le.classes_.tolist(),
    'target_mapping': dict(zip(range(len(le.classes_)), le.classes_)),
    'timestamp': timestamp
}

with open('models/shap_explainers/feature_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=4)

# ======================================================
# 7. Guardar resultados en JSON y CSV
# ======================================================

results_path_json = f"models/results_log/model_results_{timestamp}.json"
results_path_csv = f"models/results_log/model_results_{timestamp}.csv"

# JSON
with open(results_path_json, 'w') as f:
    json.dump(results, f, indent=4)

# CSV
results_df = pd.DataFrame(results).T
results_df.to_csv(results_path_csv)

print(f"\nResultados guardados en:")
print(f"  - {results_path_json}")
print(f"  - {results_path_csv}")
print(f"Explainers SHAP guardados en: models/shap_explainers/")
print("\nEntrenamiento y guardado completado.")