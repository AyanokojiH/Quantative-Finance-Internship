# Features to be designed: Apply 2 methods to achieve factor standardization. Which is:
# 1. Median method: Observe the following sequences:
#    xm: Median of sequence xi;
#    Dmad: Median of sequence |xi-xm|;
#   We adjust xi into \xi, where 
#    \xi = xm + n*Dmad, if xi > xm + n*Dmad
#    \xi = xm - n*Dmad, if xi < xm + n*Dmad
#     xi, else
#   After applying adjustment, we re-adjust \xi into ~xi = (\xi - miu)/ sigma;
#
# 2. Ranking method: 
#   We ignore xi, and, adjust every xi into it's ranking within all sequence,
#   we re-adjust xi into \xi into ~xi = (\xi - miu)/ sigma, where \xi\ stands for its ranking.
#
# After standardizing, we add a new column for every factor
# (5DAY-RETURN, NEXT_DAY_RETURN_RATIO, NEXT_5DAY_RETURN_RATIO,NEXT_20DAY_RETURN_RATIO) in the original data.



import pandas as pd
import numpy as np
from pathlib import Path

class FactorStandardizer:
    def __init__(self, input_path, output_path,n, factor ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.n = n
        self.data = None
        self.factors = [
            f'{factor}',
            # 'NEXT_DAY_RETURN_RATIO',
            # 'NEXT_5DAY_RETURN_RATIO',
            # 'NEXT_20DAY_RETURN_RATIO'
        ]

    def load_data(self):
        """Load processed data with MultiIndex (STOCK_CODE, TRADE_DATE)"""
        self.data = pd.read_parquet(self.input_path)
        print(f"Loaded data shape: {self.data.shape}")

    @staticmethod
    def median_method(series, n=3):
        arr = series.values
        xm = np.median(arr)
        dmad = np.median(np.abs(arr - xm))
        
        upper_bound = xm + n * dmad
        lower_bound = xm - n * dmad
        adjusted = np.clip(arr, lower_bound, upper_bound)
        
        standardized = (adjusted - np.mean(adjusted)) / np.std(adjusted)
        return pd.Series(standardized, index=series.index)

    @staticmethod
    def rank_method(series):
        ranks = series.rank()
        standardized = (ranks - ranks.mean()) / ranks.std()
        return standardized
    
    def standardize_factors(self):
        """Apply both standardization methods to all factors"""
        for factor in self.factors:
            if factor not in self.data.columns:
                print(f"Warning: Factor {factor} not found in data")
                continue
                
            # Median method
            self.data[f"{factor}.median_std"] = self.data.groupby('TRADE_DATE')[factor].transform(
                lambda x: self.median_method(x, self.n)
            )
            
            # Rank method
            self.data[f"{factor}.rank_std"] = self.data.groupby('TRADE_DATE')[factor].transform(
                lambda x: self.rank_method(x)
            )
        
        print("Standardization completed. New columns:")
        print([col for col in self.data.columns if '.median_std' in col or '.rank_std' in col])

    def save_results(self):
        """Save standardized data"""
        self.data.to_parquet(self.output_path)
        print(f"Saved standardized data to {self.output_path}")

    def run(self):
        """Execute full standardization pipeline"""
        self.load_data()
        self.standardize_factors()
        self.save_results()

def execute(input_path, output_path,factor, n=3):
    standardizer = FactorStandardizer(input_path, output_path, n,factor=factor)
    standardizer.run()