import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ======================================================
# 1. Load data
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
# 3. Train/test split + scaling
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'ml/models/scaler_rf.pkl')

# ======================================================
# 4. RandomForest model
# ======================================================
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

print("🔹 Training RandomForest...")
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"✅ RandomForest accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['FALSE POSITIVE', 'CONFIRMED']))

joblib.dump(rf_model, 'ml/models/randomforest_model.pkl')

# ======================================================
# 5. SHAP plots
# ======================================================
shap_folder = 'ml/models/shap_rf_plots'
os.makedirs(shap_folder, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 14

explainer = shap.Explainer(rf_model, X_train_scaled, feature_names=features)
shap_values = explainer(X_test_scaled)

# For binary classification, take positive class (CONFIRMED)
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

# Only plot if we have multiple samples
if shap_values_to_plot.shape[0] > 1:
    # Summary plot
    plt.figure()
    shap.plots.beeswarm(shap_values_to_plot, max_display=15, show=False)
    plt.title("RandomForest - SHAP Summary", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_folder, "randomforest_shap_summary.png"), dpi=300)
    plt.close()

    # Feature importance
    plt.figure()
    shap.plots.bar(shap_values_to_plot, max_display=15, show=False)
    plt.title("RandomForest - SHAP Feature Importance", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_folder, "randomforest_shap_bar.png"), dpi=300)
    plt.close()
else:
    print("⚠️ Not enough samples for SHAP plotting")

print(f"\nSHAP plots saved in {shap_folder}")
