import os
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARQUET_PATH = os.path.join(
    REPO_ROOT,
    "runs",
    "hypersteer-gemma2b-1000",
    "inference",
    "steering_data.parquet",
)

df = pd.read_parquet(PARQUET_PATH)
print(df.columns)
print(df.head())