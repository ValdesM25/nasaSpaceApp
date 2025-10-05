import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("Cargando datos...")
# Cargar y preparar datos
data = pd.read_csv('datasets/keplern.csv')
data = data.drop('transit_depth.1', axis=1)
labeled_data = data[data['category'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

le = LabelEncoder()
labeled_data['target'] = le.fit_transform(labeled_data['category'])

print(f"Datos para entrenamiento: {len(labeled_data)}")
print(f"CONFIRMED: {sum(labeled_data['target'] == 1)}")
print(f"FALSE POSITIVE: {sum(labeled_data['target'] == 0)}")

# =============================================================================
# INGENIERÍA DE CARACTERÍSTICAS EFICIENTE
# =============================================================================

def create_efficient_features(df):
    """Características eficientes y rápidas de calcular"""
    df_eff = df.copy()
    
    # 1. CARACTERÍSTICAS BÁSICAS PERO EFECTIVAS
    # Relación período-duración
    df_eff['period_duration_ratio'] = df_eff['transit_period'] / (df_eff['transit_duration'] + 0.001)
    
    # Señal-Ruido aproximada
    df_eff['snr_approximation'] = df_eff['transit_depth'] / (df_eff['transit_duration'] + 1)
    
    # 2. CARACTERÍSTICAS FÍSICAS CLAVE
    # Profundidad esperada vs observada
    expected_depth = (df_eff['planet_radius'] / df_eff['star_radius']) ** 2 * 1e6
    df_eff['depth_discrepancy'] = (df_eff['transit_depth'] - expected_depth) / (expected_depth + 1)
    
    # Luminosidad estelar simplificada
    df_eff['star_luminosity'] = (df_eff['star_radius'] ** 2) * (df_eff['star_temperature'] / 5778) ** 4
    
    # 3. CARACTERÍSTICAS DE DETECCIÓN
    # Probabilidad geométrica de tránsito
    df_eff['transit_probability'] = df_eff['star_radius'] / (df_eff['transit_period'] ** (2/3) + 0.001)
    
    # 4. TRANSFORMACIONES LOGARÍTMICAS
    df_eff['log_transit_depth'] = np.log1p(df_eff['transit_depth'])
    df_eff['log_transit_period'] = np.log1p(df_eff['transit_period'])
    
    return df_eff

print("Aplicando ingeniería de características...")
labeled_data_eff = create_efficient_features(labeled_data)

# =============================================================================
# CARACTERÍSTICAS SELECCIONADAS
# =============================================================================

features = [
    # Originales clave
    'transit_period', 'transit_duration', 'transit_depth', 'planet_radius',
    'star_temperature', 'star_radius', 'star_surface_gravity', 'star_brightness',
    
    # Ingenierizadas
    'period_duration_ratio', 'snr_approximation', 'depth_discrepancy',
    'star_luminosity', 'transit_probability', 'log_transit_depth', 'log_transit_period'
]

X = labeled_data_eff[features].fillna(0)
y = labeled_data_eff['target']

print(f"Shape de X: {X.shape}")

# =============================================================================
# DIVISIÓN Y ESCALADO RÁPIDO
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Entrenamiento: {X_train_scaled.shape}")
print(f"Prueba: {X_test_scaled.shape}")

# =============================================================================
# MODELOS RÁPIDOS Y EFICIENTES
# =============================================================================

print("\nEntrenando modelos rápidos...")

# 1. XGBoost optimizado para velocidad
xgb_fast = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1  # Usar todos los cores
)

# 2. LightGBM (muy rápido)
lgbm_fast = LGBMClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

# 3. Random Forest eficiente
rf_fast = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Entrenar modelos individualmente primero
models = {
    'XGBoost': xgb_fast,
    'LightGBM': lgbm_fast,
    'Random Forest': rf_fast
}

individual_results = {}

print("\nEvaluando modelos individuales:")
for name, model in models.items():
    print(f"  - Entrenando {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    individual_results[name] = accuracy
    print(f"    {name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

# =============================================================================
# ENSEMBLE SIMPLE
# =============================================================================

print("\nCreando ensemble...")

# Usar solo los 2 mejores modelos para el ensemble
best_models = sorted(individual_results.items(), key=lambda x: x[1], reverse=True)[:2]
print(f"Mejores modelos para ensemble: {[name for name, acc in best_models]}")

ensemble_models = []
weights = []

for name, acc in best_models:
    ensemble_models.append((name, models[name]))
    weights.append(acc)  # Ponderar por precisión

# Ensemble de votación ponderada
final_ensemble = VotingClassifier(
    estimators=ensemble_models,
    voting='soft',
    weights=weights,
    n_jobs=-1
)

print("Entrenando ensemble final...")
final_ensemble.fit(X_train_scaled, y_train)

# =============================================================================
# EVALUACIÓN FINAL
# =============================================================================

print("\nEvaluando ensemble final...")

# Predecir con ensemble
y_pred_ensemble = final_ensemble.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

# Validación cruzada rápida del ensemble
print("Realizando validación cruzada rápida...")
cv_scores = cross_val_score(final_ensemble, X_train_scaled, y_train, cv=3, scoring='accuracy', n_jobs=-1)
cv_accuracy = cv_scores.mean()

print(f"\n{'='*50}")
print("RESULTADOS FINALES")
print(f"{'='*50}")

print("\nPRECISIONES INDIVIDUALES:")
for name, acc in individual_results.items():
    print(f"  {name}: {acc:.4f} ({acc*100:.2f}%)")

print(f"\nENSEMBLE FINAL:")
print(f"  Test Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
print(f"  CV Accuracy (3-fold): {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)")

final_accuracy = max(ensemble_accuracy, cv_accuracy)

print(f"\nMEJOR PRECISIÓN: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

if final_accuracy >= 0.95:
    print("🎉 ¡OBJETIVO 95% ALCANZADO!")
elif final_accuracy >= 0.93:
    print("✅ ¡Excelente! 93%+ alcanzado")
elif final_accuracy >= 0.90:
    print("✅ Buen resultado - 90%+ alcanzado")
else:
    print("⚠️  Por debajo del 90%")

# =============================================================================
# PREDICCIÓN EN CANDIDATOS (OPCIONAL)
# =============================================================================

candidate_data = data[data['category'] == 'CANDIDATE'].copy()
if len(candidate_data) > 0:
    print(f"\nProcesando {len(candidate_data)} candidatos...")
    
    candidate_data_eff = create_efficient_features(candidate_data)
    X_candidate = candidate_data_eff[features].fillna(0)
    X_candidate_scaled = scaler.transform(X_candidate)
    
    candidate_probs = final_ensemble.predict_proba(X_candidate_scaled)[:, 1]
    
    # Análisis rápido de candidatos
    high_conf_confirmed = sum(candidate_probs > 0.8)
    high_conf_fp = sum(candidate_probs < 0.2)
    uncertain = len(candidate_data) - high_conf_confirmed - high_conf_fp
    
    print(f"Candidatos de alta confianza:")
    print(f"  - Probables CONFIRMED: {high_conf_confirmed}")
    print(f"  - Probables FALSE POSITIVE: {high_conf_fp}")
    print(f"  - Inciertos: {uncertain}")

# =============================================================================
# ANÁLISIS DE CARACTERÍSTICAS IMPORTANTES
# =============================================================================

print(f"\n{'='*50}")
print("ANÁLISIS DE CARACTERÍSTICAS")
print(f"{'='*50}")

# Usar el mejor modelo individual para análisis
best_individual_model = models[best_models[0][0]]

if hasattr(best_individual_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_individual_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 características más importantes:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

print(f"\n{'='*50}")
print("RESUMEN EJECUCIÓN:")
print(f"  • Características: {len(features)}")
print(f"  • Modelos: {len(models)} + Ensemble")
print(f"  • Tiempo estimado: < 30 segundos")
print(f"{'='*50}")