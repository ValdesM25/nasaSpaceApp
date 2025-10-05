import os
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
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

# ======================================================
# 2. Feature engineering
# ======================================================

def create_efficient_features(df):
    df_eff = df.copy()
    df_eff['period_duration_ratio'] = df_eff['transit_period'] / (df_eff['transit_duration'] + 0.001)
    df_eff['snr_approximation'] = df_eff['transit_depth'] / (df_eff['transit_duration'] + 1)
    expected_depth = (df_eff['planet_radius'] / df_eff['star_radius']) ** 2 * 1e6
    df_eff['depth_discrepancy'] = (df_eff['transit_depth'] - expected_depth) / (expected_depth + 1)
    df_eff['star_luminosity'] = (df_eff['star_radius'] ** 2) * (df_eff['star_temperature'] / 5778) ** 4
    df_eff['transit_probability'] = df_eff['star_radius'] / (df_eff['transit_period'] ** (2/3) + 0.001)
    df_eff['log_transit_depth'] = np.log1p(df_eff['transit_depth'])
    df_eff['log_transit_period'] = np.log1p(df_eff['transit_period'])
    return df_eff

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

joblib.dump(scaler, 'ml/models/scaler.pkl')

# ======================================================
# 4. Definir modelos
# ======================================================

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
# 5. Crear carpeta para gráficos SHAP
# ======================================================

shap_folder = 'ml/models/shap_plots'
os.makedirs(shap_folder, exist_ok=True)

# ======================================================
# 6. Entrenamiento + evaluación + SHAP + guardado
# ======================================================

results = {}
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

sns.set_style("whitegrid")  # Estilo seaborn
plt.rcParams["figure.figsize"] = (12, 8)  # Tamaño de gráficos grande
plt.rcParams["font.size"] = 14

for name, model in models.items():
    print(f"\n🔹 Entrenando {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Guardar modelo
    model_path = f"ml/models/{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_path)

    # Generar classification report
    report = classification_report(
        y_test, y_pred,
        target_names=['FALSE POSITIVE', 'CONFIRMED'],
        output_dict=True
    )

    results[name] = {
        "accuracy": round(acc, 4),
        "accuracy_percent": round(acc*100, 2),
        "model_path": model_path,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "classification_report": report,
        "timestamp": timestamp
    }

    print(f"✅ {name} guardado ({acc*100:.2f}% accuracy)")
    print(classification_report(y_test, y_pred, target_names=['FALSE POSITIVE', 'CONFIRMED']))

    # ======================================================
    # Generar gráficos SHAP con nombres reales de features
    # ======================================================
    explainer = shap.Explainer(model, X_train_scaled, feature_names=features)
    shap_values_all = explainer(X_test_scaled, check_additivity=False)

    # Handle binary classifiers: get the positive class Explanation
    if isinstance(shap_values_all, list):
        # shap_values_all[1] = Explanation for class 1
        shap_values_to_plot = shap_values_all[1]
    else:
        shap_values_to_plot = shap_values_all

    # Ensure we have multiple samples
    if shap_values_to_plot.shape[0] < 2:
        print(f"⚠️ Skipping SHAP plot for {name}: not enough samples")
    else:
        # Beeswarm plot
        plt.figure()
        shap.plots.beeswarm(shap_values_to_plot, max_display=15, show=False)
        plt.title(f"{name} - SHAP Summary", fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_folder, f"{name}_shap_summary.png"), dpi=300)
        plt.close()

        # Feature importance (bar plot)
        plt.figure()
        shap.plots.bar(shap_values_to_plot, max_display=15, show=False)
        plt.title(f"Random Forest - SHAP Feature Importance", fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_folder, f"{name}_shap_bar.png"), dpi=300)
        plt.close()



# ======================================================
# 7. Guardar resultados en JSON y CSV
# ======================================================

results_path_json = f"ml/models/model_results_{timestamp}.json"
results_path_csv = f"ml/models/model_results_{timestamp}.csv"

with open(results_path_json, 'w') as f:
    json.dump(results, f, indent=4)

results_df = pd.DataFrame(results).T
results_df.to_csv(results_path_csv)

print(f"\nResultados y gráficos guardados en:\n  - {results_path_json}\n  - {results_path_csv}\n  - {shap_folder}")
