import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from columnas import *

# ============================
# Carga y preparación de datos
# ============================
kepler = pd.read_csv('datasets/keplern.csv')
tess = pd.read_csv('datasets/tessn.csv')
k2 = pd.read_csv('datasets/k2n.csv')

combined = pd.concat([kepler, tess, k2])

# Solo registros que no son "CANDIDATE"
df = combined.query('category != "CANDIDATE"')

# Encode label binaria
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])  # 0 = no planeta, 1 = planeta

# Features y labels
features = [
    'transit_period',
    'transit_duration',
    'transit_depth',
    'planet_radius',
    'star_temperature',
    'star_radius',
    'star_surface_gravity',
    'star_brightness'
]

X = df[features].values
y = df['category'].values

# Normalización
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# ============================
# Balance de clases (para modelos que soportan class_weight)
# ============================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# ============================
# Definir modelos a probar
# ============================
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# ============================
# Entrenar y evaluar cada modelo
# ============================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.show()
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1
    })

# ============================
# Resultados comparativos
# ============================
results_df = pd.DataFrame(results).sort_values(by='F1-score', ascending=False)
print("🏆 Comparativa de modelos por F1-score:")
print(results_df)

# Usar copia para evitar SettingWithCopyWarning
df = combined.query('category != "CANDIDATE"').copy()

# Encode label binaria
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# Features y labels
features = [
    'transit_period',
    'transit_duration',
    'transit_depth',
    'planet_radius',
    'star_temperature',
    'star_radius',
    'star_surface_gravity',
    'star_brightness'
]

X = df[features].values
y = df['category'].values.astype(int)  # <-- asegurar que sea int

# Normalización
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Clasificadores
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
svm = SVC(probability=True, class_weight='balanced', random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('svm', svm), ('knn', knn)],
    voting='soft'  # soft para usar probabilidades
)

# Entrenar
voting_clf.fit(X_train, y_train)

# Predecir y evaluar
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
y_pred = voting_clf.predict(X_test)

print("🏆 Voting Classifier - Resultados")
print(classification_report(y_test, y_pred, target_names=le.classes_))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_)
import matplotlib.pyplot as plt
plt.title("Voting Classifier - Confusion Matrix")
plt.show()
