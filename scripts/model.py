import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DATA = PROJECT_ROOT / "data" / "processed" / "corvus_features.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_DATA)

candidate_features = [
    "decimalLatitude",
    "decimalLongitude",
    "latitude",
    "longitude",
    "year",
    "month",
    "day_of_week",
    "day_of_year",
    "total_observations",
    "speciesgroup_observations"
]

features = [col for col in candidate_features if col in df.columns]

if len(features) < 4:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    features = [
        col for col in numeric_cols
        if col not in ["target_class"]
    ]

X = df[features].fillna(0)
y = df["target_class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

with open(TABLES_DIR / "classification_report.txt", "w") as f:
    f.write(report)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix.png")
plt.close()

importance = pd.Series(model.feature_importances_, index=features)
importance = importance.sort_values(ascending=False)

importance.to_csv(TABLES_DIR / "feature_importance.csv")

plt.figure(figsize=(8, 5))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance.png")
plt.close()

print("Features used:", features)
print("Results saved.")