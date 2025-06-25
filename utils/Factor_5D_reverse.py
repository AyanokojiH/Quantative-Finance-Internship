import pandas as pd

def calculate_5d_return(df: pd.DataFrame, 
                      price_col: str = 'VWAP_PRICE_2',
                      group_col: str = 'STOCK_CODE') -> pd.Series:
    return df.groupby(group_col)[price_col].transform(
        lambda x: x.pct_change(5)
    ).rename('5D_RETURN')