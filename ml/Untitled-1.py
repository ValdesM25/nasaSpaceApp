# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from scipy.stats import randint

# %%
from columnas import *

# %%
kepler = pd.read_csv('datasets/keplern.csv')
tess = pd.read_csv('datasets/tessn.csv')
k2 = pd.read_csv('datasets/k2n.csv')

# %%
combined = pd.concat([kepler, tess, k2])

# %%
combined_no_candidates = combined.query('category != \'CANDIDATE\'')
combined_candidates = combined.query('category == \'CANDIDATE\'')

# %%
df = combined_no_candidates

# Encode label
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# Split features (X) and labels (y)
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

X = df[features]
y = df['category']

# Split training and test data 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# %%
#df['category'].value_count()

# %%
# Train the model
rf = RandomForestClassifier(
    n_estimators=100        # trees  
)
rf.fit(X_train, y_train)

# %%
# Evaluate thhe model
y_pred = rf.predict(X_test)

print("🌳 Random Forest Results")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# %%
# Confusion matrix
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, display_labels=le.classes_)
plt.title("Random Forest - Confusion Matrix")
plt.show()

# %%
# Feature importance
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

importances.plot(kind='barh', title='Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.show()

# %%



