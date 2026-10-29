import pandas as pd
from pathlib import Path

for file in Path("data").glob("*.csv"):
    df = pd.read_csv(file)

    print("\nDataset:", file.name)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head().to_string())
    print("\nMissing values:")
    print(df.isnull().sum())