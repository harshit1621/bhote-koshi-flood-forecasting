print("Script started")

from pathlib import Path
import pandas as pd

folder = Path("data")
print("Data folder exists:", folder.exists())

files = list(folder.glob("*.csv"))
print("CSV files found:", files)

for file in files:
    df = pd.read_csv(file)
    print("\nDataset:", file.name)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head().to_string())