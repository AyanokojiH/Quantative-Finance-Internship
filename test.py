import pandas as pd
import time
datasource = "./dataset/intern_data_new/basic_data/000002.parquet"

df = pd.read_parquet(datasource)

print(df.shape)
print(df.columns)
print(df.head())



