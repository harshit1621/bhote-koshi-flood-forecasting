import pandas as pd
from pathlib import Path

data_folder = Path("data")

for file in data_folder.glob("*.csv"):
    df = pd.read_csv(file)

    print("\nDataset:", file.name)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nMissing values:")
    print(df.isnull().sum())