import pandas as pd
import time
datasource = "./dataset/processed_data/neutralization.parquet"

df = pd.read_parquet(datasource)

print(df.shape)
print(df.columns)
print(df.iloc[:, 49:52].head())



