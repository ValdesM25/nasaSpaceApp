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
from ml.util.features import create_efficient_features
import warnings
warnings.filterwarnings('ignore')

# ======================================================
# 1. Cargar datos
# ======================================================

data = pd.read_csv('ml/datasets/final/keplern.csv')
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
joblib.dump(scaler, 'ml/models/scaler.pkl')

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

for name, model in models.items():
    print(f"\n🔹 Entrenando {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Guardar modelo
    model_path = f"ml/models/{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_path)

    # Guardar resultado
    results[name] = {
        "accuracy": round(acc, 4),
        "accuracy_percent": round(acc * 100, 2),
        "model_path": model_path,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "timestamp": timestamp
    }

    print(f"✅ {name} guardado ({acc*100:.2f}% accuracy)")

# ======================================================
# 6. Guardar resultados en JSON y CSV
# ======================================================

results_path_json = f"ml/models/results_log/model_results_{timestamp}.json"
results_path_csv = f"ml/models/results_log/model_results_{timestamp}.csv"

# JSON
with open(results_path_json, 'w') as f:
    json.dump(results, f, indent=4)

# CSV
results_df = pd.DataFrame(results).T
results_df.to_csv(results_path_csv)

print(f"\nResultados guardados en:\n  - {results_path_json}\n  - {results_path_csv}")
print("\nEntrenamiento y guardado completado.")
