import pandas as pd

Datasource = "./dataset/intern_data_new/basic_data/000001.parquet"

df = pd.read_parquet(Datasource)

print(df.columns)
print(df.head(10))