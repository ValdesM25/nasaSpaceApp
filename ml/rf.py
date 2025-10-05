import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
data = pd.read_csv('datasets/keplern.csv')

# Eliminar columna redundante
data = data.drop('transit_depth.1', axis=1)

# Filtrar solo CONFIRMED vs FALSE POSITIVE
labeled_data = data[data['category'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

# Codificar la variable objetivo
le = LabelEncoder()
labeled_data['target'] = le.fit_transform(labeled_data['category'])
# CONFIRMED = 1, FALSE POSITIVE = 0

print(f"Datos etiquetados: {len(labeled_data)} registros")
print(f"CONFIRMED: {sum(labeled_data['target'] == 1)}")
print(f"FALSE POSITIVE: {sum(labeled_data['target'] == 0)}")

# =============================================================================
# INGENIERÍA DE CARACTERÍSTICAS AVANZADA
# =============================================================================

def create_advanced_features(df):
    """
    Crear características avanzadas basadas en astrofísica y relaciones físicas
    """
    df_eng = df.copy()
    
    # 1. CARACTERÍSTICAS DE TRÁNSITO
    # Relación período-duración (indica geometría orbital)
    df_eng['period_duration_ratio'] = df_eng['transit_period'] / df_eng['transit_duration']
    
    # Frecuencia orbital
    df_eng['orbital_frequency'] = 1 / df_eng['transit_period']
    
    # Profundidad normalizada por duración
    df_eng['depth_duration_ratio'] = df_eng['transit_depth'] / df_eng['transit_duration']
    
    # 2. CARACTERÍSTICAS ESTELARES AVANZADAS
    # Luminosidad estimada (L ∝ R²T⁴)
    df_eng['star_luminosity'] = (df_eng['star_radius'] ** 2) * (df_eng['star_temperature'] / 5778) ** 4
    
    # Masa estelar estimada (relación masa-radio aproximada)
    df_eng['star_mass_est'] = df_eng['star_radius'] ** (1/0.8)  # Relación empírica
    
    # Edad estelar aproximada (relación gravedad-masa)
    df_eng['star_age_est'] = 10 ** (df_eng['star_surface_gravity'] - 4.44) / (df_eng['star_mass_est'] ** 2.5)
    
    # 3. CARACTERÍSTICAS PLANETARIAS AVANZADAS
    # Radio planetario normalizado por radio estelar
    df_eng['planet_star_radius_ratio'] = df_eng['planet_radius'] / df_eng['star_radius']
    
    # Profundidad teórica vs observada (debería ser ~ (Rp/Rs)²)
    df_eng['expected_depth'] = (df_eng['planet_star_radius_ratio'] ** 2) * 1e6
    df_eng['depth_discrepancy'] = (df_eng['transit_depth'] - df_eng['expected_depth']) / df_eng['expected_depth']
    
    # 4. CARACTERÍSTICAS ORBITALES
    # Semi-eje mayor estimado (Ley de Kepler III)
    df_eng['semi_major_axis'] = (df_eng['star_mass_est'] * df_eng['transit_period'] ** 2) ** (1/3)
    
    # Zona habitable interna y externa (simplificada)
    df_eng['inner_habitable_zone'] = np.sqrt(df_eng['star_luminosity'] / 1.1)
    df_eng['outer_habitable_zone'] = np.sqrt(df_eng['star_luminosity'] / 0.53)
    df_eng['in_habitable_zone'] = ((df_eng['semi_major_axis'] >= df_eng['inner_habitable_zone']) & 
                                 (df_eng['semi_major_axis'] <= df_eng['outer_habitable_zone'])).astype(int)
    
    # 5. CARACTERÍSTICAS DE DETECCIÓN
    # Señal-ruido aproximada
    df_eng['detection_snr'] = df_eng['transit_depth'] / (df_eng['transit_depth'] ** 0.5 + 100)
    
    # Probabilidad geométrica de tránsito
    df_eng['transit_probability'] = df_eng['star_radius'] / df_eng['semi_major_axis']
    
    # 6. INTERACCIONES NO LINEALES
    # Interacción temperatura-radio
    df_eng['temp_radius_interaction'] = df_eng['star_temperature'] * df_eng['star_radius']
    
    # Interacción período-profundo
    df_eng['period_depth_interaction'] = df_eng['transit_period'] * df_eng['transit_depth']
    
    # 7. CARACTERÍSTICAS ESTADÍSTICAS
    # Logaritmos de características con amplio rango
    df_eng['log_transit_depth'] = np.log10(df_eng['transit_depth'] + 1)
    df_eng['log_transit_period'] = np.log10(df_eng['transit_period'] + 1)
    df_eng['log_planet_radius'] = np.log10(df_eng['planet_radius'] + 1)
    
    # 8. CLASIFICACIONES ESTELARES
    # Tipo espectral aproximado
    conditions = [
        df_eng['star_temperature'] >= 7500,
        (df_eng['star_temperature'] >= 6000) & (df_eng['star_temperature'] < 7500),
        (df_eng['star_temperature'] >= 5200) & (df_eng['star_temperature'] < 6000),
        (df_eng['star_temperature'] >= 3700) & (df_eng['star_temperature'] < 5200),
        df_eng['star_temperature'] < 3700
    ]
    choices = [0, 1, 2, 3, 4]  # A, F, G, K, M
    df_eng['spectral_type'] = np.select(conditions, choices, default=2)
    
    # 9. CARACTERÍSTICAS DE AGREGACIÓN (para uso futuro)
    # Densidad planetaria aproximada (usando relación empírica)
    df_eng['planet_density_est'] = df_eng['planet_radius'] ** (-1.7) * 5.5  # g/cm³ aproximado
    
    return df_eng

# Aplicar ingeniería de características
labeled_data_eng = create_advanced_features(labeled_data)

# Seleccionar características finales (eliminar las originales redundantes)
feature_columns = [
    # Características originales importantes
    'transit_period', 'transit_duration', 'transit_depth', 
    'planet_radius', 'star_temperature', 'star_radius',
    'star_surface_gravity', 'star_brightness',
    
    # Características ingenierizadas
    'period_duration_ratio', 'orbital_frequency', 'depth_duration_ratio',
    'star_luminosity', 'star_mass_est', 'star_age_est',
    'planet_star_radius_ratio', 'depth_discrepancy', 'semi_major_axis',
    'in_habitable_zone', 'detection_snr', 'transit_probability',
    'temp_radius_interaction', 'period_depth_interaction',
    'log_transit_depth', 'log_transit_period', 'log_planet_radius',
    'spectral_type', 'planet_density_est'
]

X = labeled_data_eng[feature_columns]
y = labeled_data_eng['target']

print(f"Características finales: {len(feature_columns)}")
print(f"Forma de X: {X.shape}")

# =============================================================================
# PREPROCESAMIENTO ROBUSTO
# =============================================================================

# Manejar valores infinitos y NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Entrenamiento: {X_train_scaled.shape[0]} muestras")
print(f"Prueba: {X_test_scaled.shape[0]} muestras")

# =============================================================================
# MODELADO AVANZADO
# =============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluar modelo de manera completa"""
    # Entrenar
    model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"\n{'='*50}")
    print(f"MODELO: {model_name}")
    print(f"{'='*50}")
    print(f"Precisión: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Validación Cruzada (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['FALSE POSITIVE', 'CONFIRMED']))
    
    return accuracy, cv_scores.mean()

# 1. XGBoost con ajuste fino
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # Balancear clases
)

# 2. Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# 3. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

# Evaluar todos los modelos
models = {
    'XGBoost': xgb_model,
    'Gradient Boosting': gb_model,
    'Random Forest': rf_model
}

results = {}
for name, model in models.items():
    acc, cv_acc = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
    results[name] = {'accuracy': acc, 'cv_accuracy': cv_acc}

# =============================================================================
# ENSAMBLE FINAL Y OPTIMIZACIÓN
# =============================================================================

print(f"\n{'='*60}")
print("OPTIMIZACIÓN FINAL CON EL MEJOR MODELO")
print(f"{'='*60}")

# Seleccionar el mejor modelo
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]

print(f"Mejor modelo: {best_model_name}")

# Búsqueda de hiperparámetros para el mejor modelo
if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [6, 7, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05, 0.1, 0.15]
    }
else:  # Random Forest
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [8, 10, 12],
        'min_samples_split': [2, 5, 10]
    }

# Optimizar (comentado para velocidad, descomentar para mejor rendimiento)
"""
grid_search = GridSearchCV(
    best_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

best_model_optimized = grid_search.best_estimator_
print(f"Mejores parámetros: {grid_search.best_params_}")
"""

# Usar el modelo base ya que GridSearchCV puede ser lento
final_model = best_model
final_model.fit(X_train_scaled, y_train)

# Evaluación final
y_pred_final = final_model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, y_pred_final)

print(f"\n{'='*50}")
print("RESULTADO FINAL")
print(f"{'='*50}")
print(f"Modelo: {best_model_name}")
print(f"Precisión Final: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

if final_accuracy >= 0.90:
    print("🎯 ¡OBJETIVO CUMPLIDO! Precisión >90%")
else:
    print("⚠️  Precisión por debajo del 90%, considerar:")
    print("   - Más ingeniería de características")
    print("   - Optimización más extensa de hiperparámetros")
    print("   - Ensamblaje de modelos")

# =============================================================================
# ANÁLISIS DE CARACTERÍSTICAS IMPORTANTES
# =============================================================================

if hasattr(final_model, 'feature_importances_'):
    print(f"\n{'='*50}")
    print("TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES")
    print(f"{'='*50}")
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))

# =============================================================================
# PREDICCIÓN EN DATOS CANDIDATE (OPCIONAL)
# =============================================================================

# Aplicar el mismo preprocesamiento a datos CANDIDATE
candidate_data = data[data['category'] == 'CANDIDATE'].copy()
if len(candidate_data) > 0:
    candidate_data_eng = create_advanced_features(candidate_data)
    X_candidate = candidate_data_eng[feature_columns].fillna(X.median())
    X_candidate_scaled = scaler.transform(X_candidate)
    
    # Predecir probabilidades
    candidate_probs = final_model.predict_proba(X_candidate_scaled)[:, 1]
    candidate_data['predicted_prob_CONFIRMED'] = candidate_probs
    candidate_data['predicted_label'] = (candidate_probs > 0.5).astype(int)
    candidate_data['predicted_category'] = le.inverse_transform(candidate_data['predicted_label'])
    
    print(f"\nPredicciones para {len(candidate_data)} candidatos:")
    print(f" - Probables CONFIRMED: {sum(candidate_probs > 0.7)}")
    print(f" - Probables FALSE POSITIVE: {sum(candidate_probs < 0.3)}")
    print(f" - Inciertos: {sum((candidate_probs >= 0.3) & (candidate_probs <= 0.7))}")

print(f"\n{'='*60}")
print("INGENIERÍA DE CARACTERÍSTICAS APLICADA:")
print(f" - Características originales: 8")
print(f" - Características ingenierizadas: {len(feature_columns) - 8}")
print(f" - Total características: {len(feature_columns)}")
print(f"{'='*60}")