"""
Features to be designed: 
Regress the profit on industies to avoid influences, then regress factors on the residuals ,
as well as to test it's t,p, return rate and R-squared.
"""


import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import numpy as np
from typing import Dict, List

class FactorReturnCalculator:
    def __init__(self, data: pd.DataFrame,tested_length,factor):
        self.data = data.copy()
        self.industry_cols = self._detect_industry_columns()
        self.required_columns = [f'NEXT_{tested_length}DAY_RETURN_RATIO']
        self._validate_data()
        self.factor = factor
        self.data = self.data.dropna(subset=self.required_columns + self.industry_cols)

    def _detect_industry_columns(self) -> List[str]:
        industry_prefixes = ['Agriculture', 'Automobiles', 'Banks', 'BuildMater', 'Chemicals', 
        'Commerce', 'Computers', 'Conglomerates', 'ConstrDecor', 'Defense',
        'ElectricalEquip', 'Electronics', 'FoodBeverages', 'HealthCare',
        'HomeAppliances', 'Leisure', 'LightIndustry', 'MachineEquip', 'Media',
        'Mining', 'NonbankFinan', 'NonferrousMetals', 'RealEstate', 'Steel',
        'Telecoms', 'TextileGarment', 'Transportation', 'Utilities',
        'BasicChemicals', 'BeautyCare', 'Coal', 'EnvironProtect', 'Petroleum',
        'PowerEquip', 'RetailTrade', 'SocialServices', 'TextileApparel']  
        return [col for col in self.data.columns if any(col.startswith(prefix) for prefix in industry_prefixes)]

    def _validate_data(self) -> None:
        missing_cols = [col for col in self.required_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        if not self.industry_cols:
            raise ValueError("No industry dummy columns detected")

    def filter_by_year(self, year: int) -> None:
        if 'TRADE_DATE' in self.data.index.names:
            self.data = self.data[self.data.index.get_level_values('TRADE_DATE').year == year]
        else:
            self.data = self.data[pd.to_datetime(self.data['TRADE_DATE']).dt.year == year]

    # def neutralize_returns_by_industry(self, return_col: str) -> pd.Series:
    #    X = sm.add_constant(self.data[self.industry_cols])
    #    y = self.data[return_col]
    #    model = sm.OLS(y, X).fit()  
    #    return model.resid

    def calculate_factor_return(self, factor_col: str, return_col: str = 'NEXT_DAY_RETURN_RATIO') -> Dict[str, float]:
        # residual_return = self.neutralize_returns_by_industry(return_col)
        if factor_col not in self.data.columns or return_col not in self.data.columns:
            raise ValueError(f"Column {factor_col} or {return_col} not found in data")
        
        # 检查缺失值和无穷大
        if self.data[[factor_col, return_col]].isnull().any().any():
            print(f"NaN found in columns: {self.data[[factor_col, return_col]].columns[self.data[[factor_col, return_col]].isnull().any()].tolist()}")
        
        if np.isinf(self.data[[factor_col, return_col]]).any().any():
            print(f"Inf found in columns: {self.data[[factor_col, return_col]].columns[np.isinf(self.data[[factor_col, return_col]]).any()].tolist()}")
        
        # 过滤非法值
        valid_data = self.data[[factor_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) == 0:
            raise ValueError("No valid data after filtering nan/inf")
        
        X = sm.add_constant(valid_data[factor_col])
        Y = valid_data[return_col]
        model = sm.OLS(Y, X).fit()
        return {
            'factor': factor_col,
            'return': f"{model.params[factor_col]*100:.4f}%",
            't_value': f"{model.tvalues[factor_col]:.2f}",
            'p_value': f"{model.pvalues[factor_col]:.2f}",
            'r_squared': f"{model.rsquared*100:.4f}%",
            'n_obs': int(model.nobs),
        }

def execute(input_path,short,med,long,backtest,factor):
    df = pd.read_parquet(Path(input_path))
    calculator = FactorReturnCalculator(df,short,factor=factor)
    
    # Add year selection
    selected_year = backtest
    if selected_year != 0:
        calculator.filter_by_year(selected_year)
    
    factor_pool = [f"{factor}.rank_std_neutral",f"{factor}.median_std_neutral"]
    target_pool = [f"NEXT_{short}DAY_RETURN_RATIO",f"NEXT_{med}DAY_RETURN_RATIO",f"NEXT_{long}DAY_RETURN_RATIO"]
    
    for factor in factor_pool:
        for target in target_pool:
            result = calculator.calculate_factor_return(factor, target)
            print(f"Result for factor {factor} and target {target}: ")
            for k, v in result.items():
                print(f"{k}: {v}")
            print("="*50)
            print()