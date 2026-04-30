import pandas as pd
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "Corvus corone.csv.zip"
OUTPUT_DATA = PROJECT_ROOT / "data" / "processed" / "corvus_cleaned.csv"

with zipfile.ZipFile(RAW_DATA, "r") as z:
    print("Files inside zip:", z.namelist())
    csv_name = z.namelist()[0]

    with z.open(csv_name) as f:
        df = pd.read_csv(f)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

df = df.drop_duplicates()

date_cols = [col for col in df.columns if "date" in col.lower()]

if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
else:
    print("No date column found.")

numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

text_cols = df.select_dtypes(include=["object"]).columns
df[text_cols] = df[text_cols].fillna("Unknown")

OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_DATA, index=False)

print("Cleaned shape:", df.shape)
print("Saved cleaned data to:", OUTPUT_DATA)