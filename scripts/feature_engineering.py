import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DATA = PROJECT_ROOT / "data" / "processed" / "corvus_cleaned.csv"
OUTPUT_DATA = PROJECT_ROOT / "data" / "processed" / "corvus_features.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_DATA, low_memory=False)
date_col = [col for col in df.columns if "date" in col.lower()][0]
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

df["year"] = df[date_col].dt.year
df["month"] = df[date_col].dt.month
df["day_of_week"] = df[date_col].dt.dayofweek
df["day_of_year"] = df[date_col].dt.dayofyear

possible_targets = [
    col for col in df.columns
    if "corvus" in col.lower()
    or "species_observations" in col.lower()
    or "observations" in col.lower()
]

print("Possible target columns:", possible_targets)

target_col = possible_targets[-1]
df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
df["target_class"] = (df[target_col] > 0).astype(int)

monthly = df.groupby("month")[target_col].sum()

plt.figure(figsize=(8, 5))
monthly.plot(kind="bar")
plt.title("Corvus corone observations by month")
plt.xlabel("Month")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "observations_by_month.png")
plt.close()

df["target_class"].value_counts().plot(kind="bar")
plt.title("Class balance: observed vs not observed")
plt.xlabel("Target class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "class_balance.png")
plt.close()

df.to_csv(OUTPUT_DATA, index=False)

print("Target column used:", target_col)
print("Saved feature data to:", OUTPUT_DATA)
print(df.head())