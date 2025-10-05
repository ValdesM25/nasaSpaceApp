import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

kepler = pd.read_csv('datasets/kepler.csv')
k2n = pd.read_csv('datasets/k2n.csv')
tessn = pd.read_csv('datasets/tessn.csv')

# Número de nulos por fila
kepler["num_nulos"] = kepler.isnull().sum(axis=1)
k2n["num_nulos"] = k2n.isnull().sum(axis=1)
tessn["num_nulos"] = tessn.isnull().sum(axis=1)

# Proporción de nulos por fila
kepler["prop_nulos"] = kepler.isnull().mean(axis=1)
k2n["prop_nulos"] = k2n.isnull().mean(axis=1)
tessn["prop_nulos"] = tessn.isnull().mean(axis=1)

# Definir umbral: eliminar registros con más del 20% de valores faltantes
umbral = 0.20

kepler = kepler[kepler["prop_nulos"] <= umbral].copy()
k2n = k2n[k2n["prop_nulos"] <= umbral].copy()
tessn = tessn[tessn["prop_nulos"] <= umbral].copy()

# 1. Identificar columnas numéricas para imputación
def get_numeric_columns(df):
    return df.select_dtypes(include=['number']).columns.tolist()

# 2. Función para aplicar KNN Imputer
def aplicar_knn_imputer(df, n_neighbors=5):
    # Hacer copia para no modificar el original
    df_imputed = df.copy()
    
    # Identificar columnas numéricas
    numeric_cols = get_numeric_columns(df)
    
    if numeric_cols:
        # Separar columnas numéricas y categóricas
        numeric_data = df[numeric_cols]
        
        # Escalar los datos para KNN (importante para que todas las variables tengan igual peso)
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_data)
        
        # Aplicar KNN Imputer
        imputer = KNNImputer(n_neighbors=n_neighbors)
        numeric_imputed = imputer.fit_transform(numeric_scaled)
        
        # Revertir escalado
        numeric_imputed = scaler.inverse_transform(numeric_imputed)
        
        # Reemplazar columnas numéricas en el DataFrame
        df_imputed[numeric_cols] = numeric_imputed
    
    return df_imputed

# 3. Aplicar KNN Imputer a cada dataset
print("Aplicando KNN Imputer...")
kepler_imputed = aplicar_knn_imputer(kepler)
k2n_imputed = aplicar_knn_imputer(k2n)
tessn_imputed = aplicar_knn_imputer(tessn)

# 4. Verificar resultados
print("\nValores nulos antes de imputación:")
print(f"Kepler: {kepler.isnull().sum().sum()}")
print(f"K2: {k2n.isnull().sum().sum()}")
print(f"TESS: {tessn.isnull().sum().sum()}")

print("\nValores nulos después de imputación:")
print(f"Kepler: {kepler_imputed.isnull().sum().sum()}")
print(f"K2: {k2n_imputed.isnull().sum().sum()}")
print(f"TESS: {tessn_imputed.isnull().sum().sum()}")

# Contar no candidatos en cada dataset
no_candidatos_kepler = (kepler['category'] != 'CANDIDATE').sum()
no_candidatos_k2n = (k2n['category'] != 'CANDIDATE').sum()
no_candidatos_tessn = (tessn['category'] != 'CANDIDATE').sum()

# Total de no candidatos
total_no_candidatos = no_candidatos_kepler + no_candidatos_k2n + no_candidatos_tessn

print(f"No candidatos en Kepler: {no_candidatos_kepler}")
print(f"No candidatos en K2: {no_candidatos_k2n}")
print(f"No candidatos en TESS: {no_candidatos_tessn}")
print(f"Total de no candidatos: {total_no_candidatos}")

kepler = kepler.drop(columns=["num_nulos", "prop_nulos"])
k2n = k2n.drop(columns=["num_nulos", "prop_nulos"])
tessn = tessn.drop(columns=["num_nulos", "prop_nulos"])

kepler.to_csv('datasets/keplern.csv', index=False)
k2n.to_csv('datasets/k2n.csv', index=False)
tessn.to_csv('datasets/tessn.csv', index=False)


