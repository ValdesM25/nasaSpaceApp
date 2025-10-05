import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("datasets/combined.csv")

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
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
rf = RandomForestClassifier(
    n_estimators=300,        # trees
    max_depth=None,          # depth limit
    min_samples_split=2,     # minimum samples to divide a node
    random_state=42,         # seed
    n_jobs=-1,               # use all cpu cores
    class_weight='balanced'  
)

rf.fit(X_train, y_train)

# Evaluate thhe model
y_pred = rf.predict(X_test)

print("🌳 Random Forest Results")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, display_labels=le.classes_)
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Feature importance
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

importances.plot(kind='barh', title='Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.show()
