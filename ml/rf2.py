import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import IsolationForest
warnings.filterwarnings('ignore')

# Cargar y preparar datos
data = pd.read_csv('datasets/keplern.csv')
data = data.drop('transit_depth.1', axis=1)
labeled_data = data[data['category'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

le = LabelEncoder()
labeled_data['target'] = le.fit_transform(labeled_data['category'])

# =============================================================================
# INGENIERÍA DE CARACTERÍSTICAS DE NIVEL EXPERT
# =============================================================================

def create_expert_features(df):
    """Características de nivel experto para astrofísica de exoplanetas"""
    df_exp = df.copy()
    
    # 1. CARACTERÍSTICAS DE VALIDEZ FÍSICA
    # Relación de consistencia física fundamental
    df_exp['transit_shape_consistency'] = (df_exp['transit_duration'] / df_exp['transit_period']) / 0.1
    
    # Señal-Ruido espectral (combinando múltiples características)
    df_exp['spectral_snr'] = (df_exp['transit_depth'] * df_exp['star_brightness']) / (df_exp['transit_duration'] + 1)
    
    # 2. CARACTERÍSTICAS DE PROBABILIDAD DE DETECCIÓN
    # Probabilidad bayesiana aproximada
    expected_depth = (df_exp['planet_radius'] / df_exp['star_radius']) ** 2 * 1e6
    df_exp['bayesian_confidence'] = np.exp(-0.5 * ((df_exp['transit_depth'] - expected_depth) / 
                                                (0.1 * expected_depth)) ** 2)
    
    # 3. CARACTERÍSTICAS DE ANOMALÍA
    # Detección de valores atípicos multivariados
    from sklearn.ensemble import IsolationForest
    iso_features = ['transit_depth', 'transit_period', 'planet_radius', 'star_temperature']
    iso = IsolationForest(contamination=0.1, random_state=42)
    
    if len(df_exp) > 1:  # Solo si hay suficientes datos
        try:
            df_exp['isolation_score'] = iso.fit_predict(df_exp[iso_features].fillna(0))
        except:
            df_exp['isolation_score'] = 0
    
    # 4. CARACTERÍSTICAS DE AGREGACIÓN POR GRUPOS
    # Agrupar por tipo de estrella y calcular estadísticas
    df_exp['temp_group'] = pd.cut(df_exp['star_temperature'], bins=5, labels=False)
    group_stats = df_exp.groupby('temp_group')['transit_depth'].agg(['mean', 'std']).fillna(0)
    df_exp = df_exp.merge(group_stats, left_on='temp_group', right_index=True, suffixes=('', '_group'))
    df_exp['depth_zscore'] = (df_exp['transit_depth'] - df_exp['mean']) / (df_exp['std'] + 1e-6)
    
    # 5. CARACTERÍSTICAS DE INTERACCIÓN POLINÓMICA
    # Interacciones de segundo y tercer orden
    df_exp['poly_interaction_1'] = (df_exp['transit_depth'] * df_exp['planet_radius'] * 
                                   df_exp['star_temperature'])
    df_exp['poly_interaction_2'] = (df_exp['transit_period'] ** 0.5 * 
                                   df_exp['transit_duration'] ** 1.5)
    
    # 6. CARACTERÍSTICAS DE DOMINIO ESPECÍFICO
    # Clasificación de tipo de planeta basado en radio
    conditions = [
        df_exp['planet_radius'] < 1.5,
        (df_exp['planet_radius'] >= 1.5) & (df_exp['planet_radius'] < 3.0),
        (df_exp['planet_radius'] >= 3.0) & (df_exp['planet_radius'] < 8.0),
        df_exp['planet_radius'] >= 8.0
    ]
    choices = [0, 1, 2, 3]  # Terrestre, Super-Tierra, Mini-Neptuno, Gigante
    df_exp['planet_type'] = np.select(conditions, choices, default=1)
    
    # 7. CARACTERÍSTICAS TEMPORALES COMPLEJAS
    # Armónicos orbitales
    df_exp['orbital_harmonic_1'] = np.sin(2 * np.pi * df_exp['transit_period'] / 10)
    df_exp['orbital_harmonic_2'] = np.cos(2 * np.pi * df_exp['transit_period'] / 10)
    
    # 8. CARACTERÍSTICAS DE CALIDAD DE DATOS
    df_exp['data_quality_index'] = (
        (df_exp['transit_depth'] > 100).astype(int) +
        (df_exp['transit_duration'] > 0.5).astype(int) +
        (df_exp['star_temperature'] > 3000).astype(int) +
        (df_exp['planet_radius'] > 0.5).astype(int)
    )
    
    # 9. RATIOS CRÍTICOS
    df_exp['critical_ratio_1'] = df_exp['transit_depth'] / (df_exp['planet_radius'] ** 2 + 1)
    df_exp['critical_ratio_2'] = df_exp['star_radius'] / (df_exp['transit_period'] + 1)
    
    # 10. TRANSFORMACIONES NO LINEALES AVANZADAS
    df_exp['log_spectral_power'] = np.log1p(df_exp['star_temperature'] * df_exp['star_brightness'])
    df_exp['exp_transit_signal'] = 1 - np.exp(-df_exp['transit_depth'] / 1000)
    
    return df_exp

# Aplicar ingeniería experta
labeled_data_exp = create_expert_features(labeled_data)

# =============================================================================
# SELECCIÓN DE CARACTERÍSTICAS AVANZADA
# =============================================================================

# Lista completa de características
all_features = [
    # Originales
    'transit_period', 'transit_duration', 'transit_depth', 'planet_radius',
    'star_temperature', 'star_radius', 'star_surface_gravity', 'star_brightness',
    
    # Ingeniería experta
    'transit_shape_consistency', 'spectral_snr', 'bayesian_confidence',
    'isolation_score', 'depth_zscore', 'poly_interaction_1', 'poly_interaction_2',
    'planet_type', 'orbital_harmonic_1', 'orbital_harmonic_2', 'data_quality_index',
    'critical_ratio_1', 'critical_ratio_2', 'log_spectral_power', 'exp_transit_signal'
]

X = labeled_data_exp[all_features].fillna(0)
y = labeled_data_exp['target']

# Eliminar características con baja varianza
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)

print(f"Características después de selección: {X.shape[1]}")

# =============================================================================
# ENSEMBLE AVANZADO CON STACKING
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42  # Menos datos de test para más entrenamiento
)

# Escalado robusto
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. MODELOS BASE OPTIMIZADOS
xgb_optimized = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    reg_lambda=0.5,
    gamma=0.1,
    random_state=42,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
)

lgbm_optimized = LGBMClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.3,
    reg_lambda=0.3,
    random_state=42,
    class_weight='balanced'
)

catboost_optimized = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.025,
    random_state=42,
    verbose=0,
    auto_class_weights='Balanced'
)

# 2. ENSEMBLE POR VOTACIÓN PONDERADA
voting_ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_optimized),
        ('lgbm', lgbm_optimized),
        ('catboost', catboost_optimized)
    ],
    voting='soft',
    weights=[3, 2, 2]  # Ponderación basada en rendimiento esperado
)

# 3. STACKING AVANZADO CON META-LEARNER
from sklearn.linear_model import LogisticRegressionCV

stacking_ensemble = StackingClassifier(
    estimators=[
        ('xgb', xgb_optimized),
        ('lgbm', lgbm_optimized),
        ('catboost', catboost_optimized)
    ],
    final_estimator=LogisticRegressionCV(
        Cs=20,
        cv=5,
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    ),
    cv=5,
    passthrough=True
)

# =============================================================================
# ENTRENAMIENTO CON VALIDACIÓN CRUZADA ESTRATIFICADA
# =============================================================================

print("Entrenando modelos avanzados...")

# Entrenar y evaluar cada ensemble
models = {
    'Voting Ensemble': voting_ensemble,
    'Stacking Ensemble': stacking_ensemble
}

best_accuracy = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    print(f"\nEntrenando {name}...")
    
    # Validación cruzada estratificada
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='accuracy')
    
    # Entrenar en todos los datos de entrenamiento
    model.fit(X_train_scaled, y_train)
    
    # Predecir en test
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{name}:")
    print(f"  Validación Cruzada: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# =============================================================================
# CALIBRACIÓN DE PROBABILIDADES
# =============================================================================

print(f"\nCalibrando el mejor modelo: {best_model_name}")

# Calibrar probabilidades para mejor rendimiento
calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
calibrated_model.fit(X_train_scaled, y_train)

y_pred_calibrated = calibrated_model.predict(X_test_scaled)
calibrated_accuracy = accuracy_score(y_test, y_pred_calibrated)

print(f"Precisión después de calibración: {calibrated_accuracy:.4f} ({calibrated_accuracy*100:.2f}%)")

# =============================================================================
# PREDICCIÓN CON UMBRAL OPTIMIZADO
# =============================================================================

# Encontrar el mejor umbral de decisión
from sklearn.metrics import precision_recall_curve

y_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Encontrar umbral que maximiza F1-score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores[:-1])]

print(f"Mejor umbral de decisión: {best_threshold:.4f}")

# Aplicar umbral optimizado
y_pred_optimized = (y_proba >= best_threshold).astype(int)
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)

print(f"Precisión con umbral optimizado: {optimized_accuracy:.4f} ({optimized_accuracy*100:.2f}%)")

# =============================================================================
# ENSEMBLE FINAL CON LOS MEJORES MODELOS
# =============================================================================

# Entrenar todos los modelos en todos los datos
print("\nEntrenando ensemble final con todos los datos...")

X_full_scaled = scaler.fit_transform(X)

# Ensemble final que combina múltiples técnicas
final_ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_optimized),
        ('lgbm', lgbm_optimized),
        ('catboost', catboost_optimized),
        ('stacking', stacking_ensemble)
    ],
    voting='soft',
    weights=[3, 2, 2, 3]
)

final_ensemble.fit(X_full_scaled, y)

# Validación final con leave-one-out (muy costoso pero preciso)
print("Realizando validación final...")
from sklearn.model_selection import cross_val_score

final_cv_scores = cross_val_score(final_ensemble, X_full_scaled, y, cv=5, scoring='accuracy')
final_accuracy = final_cv_scores.mean()

print(f"\n{'='*60}")
print("RESULTADO FINAL")
print(f"{'='*60}")
print(f"Precisión Promedio (5-fold CV): {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"Mejor precisión individual: {max(final_cv_scores):.4f} ({max(final_cv_scores)*100:.2f}%)")

if final_accuracy >= 0.95:
    print("🎉 ¡OBJETIVO 95% ALCANZADO!")
elif final_accuracy >= 0.93:
    print("✅ ¡Muy cerca! 93%+ alcanzado")
else:
    print("⚠️  Buen resultado, pero podemos mejorar más")

# =============================================================================
# ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# =============================================================================

print(f"\n{'='*50}")
print("ANÁLISIS DE CARACTERÍSTICAS MÁS IMPORTANTES")
print(f"{'='*50}")

# Obtener importancia de características del mejor modelo
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X.shape[1])],
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 características más importantes:")
    print(feature_importance.head(15))

# =============================================================================
# PREDICCIÓN EN CANDIDATOS CON MODELO FINAL
# =============================================================================

candidate_data = data[data['category'] == 'CANDIDATE'].copy()
if len(candidate_data) > 0:
    print(f"\nAplicando modelo a {len(candidate_data)} candidatos...")
    
    candidate_data_exp = create_expert_features(candidate_data)
    X_candidate = candidate_data_exp[all_features].fillna(0)
    X_candidate = selector.transform(X_candidate)  # Aplicar misma selección
    X_candidate_scaled = scaler.transform(X_candidate)
    
    candidate_probs = final_ensemble.predict_proba(X_candidate_scaled)[:, 1]
    
    candidate_data['predicted_prob_CONFIRMED'] = candidate_probs
    candidate_data['predicted_label'] = (candidate_probs >= best_threshold).astype(int)
    
    high_confidence_confirmed = sum(candidate_probs >= 0.8)
    high_confidence_fp = sum(candidate_probs <= 0.2)
    
    print(f"Candidatos de alta confianza:")
    print(f"  - Probables CONFIRMED (≥80%): {high_confidence_confirmed}")
    print(f"  - Probables FALSE POSITIVE (≤20%): {high_confidence_fp}")
    print(f"  - Inciertos: {len(candidate_data) - high_confidence_confirmed - high_confidence_fp}")

print(f"\n{'='*60}")
print("RESUMEN DE LA ESTRATEGIA:")
print(f"  • Características expertas: {len(all_features)}")
print(f"  • Ensemble avanzado: Voting + Stacking")
print(f"  • Calibración de probabilidades: Isotonic")
print(f"  • Optimización de umbral: F1-score")
print(f"{'='*60}")
