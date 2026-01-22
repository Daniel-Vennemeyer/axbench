import pandas as pd
df = pd.read_parquet("runs/hypersteer-gemma2b-1000/inference/steering_data.parquet")
print(df.columns)
print(df.head())