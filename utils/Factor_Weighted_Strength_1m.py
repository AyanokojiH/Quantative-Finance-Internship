import numpy as np
import pandas as pd

def calculate_weighted_strength_1m(df: pd.DataFrame,
                                  price_col: str = 'CLOSE_PRICE_2',
                                  turnover_col: str = 'TURNOVER_RATE',
                                  group_col: str = 'STOCK_CODE',
                                  window: int = 20) -> pd.Series:
    # 转换为NumPy数组提升性能
    prices = df[price_col].values
    turnovers = df[turnover_col].values
    groups = df[group_col].values
    
    # 预分配结果数组
    weighted_strength = np.full_like(prices, np.nan, dtype=np.float64)
    
    # 查找每个股票组的起止位置
    group_changes = np.where(groups[1:] != groups[:-1])[0] + 1
    group_starts = np.concatenate([[0], group_changes])
    group_ends = np.concatenate([group_changes, [len(groups)]])
    
    for start, end in zip(group_starts, group_ends):
        if end - start < window:  # 数据不足一个窗口则跳过
            continue
            
        # 提取当前股票的数据块
        group_prices = prices[start:end]
        group_turnovers = turnovers[start:end]
        
        # 计算每日收益率
        daily_returns = np.zeros_like(group_prices)
        daily_returns[1:] = group_prices[1:] / group_prices[:-1] - 1
        
        # 计算滚动日均收益率（跳过前window-1个无效值）
        avg_returns = np.full_like(group_prices, np.nan)
        for i in range(window, len(group_prices)):
            avg_returns[i] =- np.mean(daily_returns[i-window+1:i+1])
        
        # 计算换手率权重（每日归一化）
        turnover_weights = np.zeros_like(group_turnovers)
        for i in range(window, len(group_turnovers)):
            window_turnovers = group_turnovers[i-window+1:i+1]
            turnover_weights[i] = group_turnovers[i] / np.sum(window_turnovers)
        
        # 计算加权强度因子
        weighted_strength[start+window-1:end] = (avg_returns[window-1:] * turnover_weights[window-1:])
    
    return pd.Series(weighted_strength, index=df.index, name='weighted_strength_1m')