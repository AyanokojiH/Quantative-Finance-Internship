import numpy as np
import pandas as pd

def calculate_relative_strength(df: pd.DataFrame,
                                  price_col: str = 'VWAP_PRICE_2',
                                  group_col: str = 'STOCK_CODE',
                                  month: int = 12) -> pd.Series:
    
    window = month*20
    
    # 转换为NumPy数组提升性能
    prices = df[price_col].values
    groups = df[group_col].values
    
    # 预分配结果数组
    momentum = np.full_like(prices, np.nan, dtype=np.float64)
    
    # 查找每个股票组的起止位置
    group_changes = np.where(groups[1:] != groups[:-1])[0] + 1
    group_starts = np.concatenate([[0], group_changes])
    group_ends = np.concatenate([group_changes, [len(groups)]])
    
    for start, end in zip(group_starts, group_ends):
        if end - start < window:  # 数据不足窗口则跳过
            continue
            
        # 计算窗口收益率: (当前价 / window日前价格) - 1
        group_prices = prices[start:end]
        momentum[start+window-1:end] = (group_prices[window-1:] / group_prices[:end-start-window+1]) - 1
    
    return pd.Series(momentum, index=df.index, name=f'relative_strength_{month}m')