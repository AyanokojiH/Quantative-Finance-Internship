import numpy as np
import pandas as pd

def calculate_5d_reverse(df: pd.DataFrame, 
                       price_col: str = 'VWAP_PRICE_2',
                       group_col: str = 'STOCK_CODE') -> pd.Series:
    prices = df[price_col].values
    groups = df[group_col].values
    
    group_changes = np.where(groups[1:] != groups[:-1])[0] + 1
    group_starts = np.concatenate([[0], group_changes])
    group_ends = np.concatenate([group_changes, [len(groups)]])
    
    returns = np.full_like(prices, np.nan, dtype=np.float64)

    for start, end in zip(group_starts, group_ends):
        if end - start > 5: 
            group_prices = prices[start:end]
            returns[start+5:end] =(group_prices[:-5] / group_prices[5:]) -1
    
    return pd.Series(returns, index=df.index, name='5D_REVERSE')


# def numpy_matrix_method(df):
#     prices = df['price'].values
#     stock_ids = df['stock'].values
    
#     # 创建分组边界索引
#     change_points = np.where(stock_ids[1:] != stock_ids[:-1])[0] + 1
#     group_starts = np.concatenate([[0], change_points])
#     group_ends = np.concatenate([change_points, [len(stock_ids)]])
    
#     returns = np.full_like(prices, np.nan, dtype=np.float64)
    
#     for start, end in zip(group_starts, group_ends):
#         if end - start > 5:
#             group_prices = prices[start:end]
#             returns[start+5:end] = group_prices[5:] / group_prices[:-5] - 1
    
#     return pd.Series(returns, index=df.index)