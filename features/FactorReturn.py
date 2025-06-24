"""
Features to be designed: 
Regress the profit on industies to avoid influences, then regress factors on the residuals ,
as well as to test it's t,p, return rate and R-squared.
"""


import pandas as pd
import statsmodels.api as sm
from typing import Dict, List

class FactorReturnCalculator:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.industry_cols = self._detect_industry_columns()
        self.required_columns = ['NEXT_DAY_RETURN_RATIO']
        
        # 预先删除缺失值
        self._validate_data()
        self.data = self.data.dropna(
            subset=self.required_columns + self.industry_cols
        )

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
            raise ValueError(f"缺失必要列: {missing_cols}")
        if not self.industry_cols:
            raise ValueError("未检测到行业哑变量列")

    def neutralize_returns_by_industry(self, return_col: str) -> pd.Series:
        X = sm.add_constant(self.data[self.industry_cols])
        y = self.data[return_col]
        model = sm.OLS(y, X).fit()  
        return model.resid

    def calculate_factor_return(self, factor_col: str, return_col: str = 'NEXT_DAY_RETURN_RATIO') -> Dict[str, float]:
        # 1. Reg Profit on industries
        residual_return = self.neutralize_returns_by_industry(return_col)
        
        # 2. Reg factor on residuals
        X = sm.add_constant(self.data[[factor_col]])
        
        model = sm.OLS(residual_return, X).fit()
        return {
            'factor': factor_col,
            'return': str(model.params[factor_col]*100) + str("%"),
            't_value': model.tvalues[factor_col],
            'p_value': model.pvalues[factor_col],
            'r_squared': str(model.rsquared*100) + str("%"),
            'n_obs': model.nobs
        }

if __name__ == "__main__":
    df = pd.read_parquet("../dataset/processed_data/neutralization.parquet")  
    calculator = FactorReturnCalculator(df)
    factor_pool = ["5D_RETURN.rank_neutral","5D_RETURN.median_neutral"]
    target_pool = ["NEXT_DAY_RETURN_RATIO","NEXT_5DAY_RETURN_RATIO","NEXT_20DAY_RETURN_RATIO"]
    for factor in factor_pool:
        for target in target_pool:
            result = calculator.calculate_factor_return(factor,target)
            print(f"Result for factor {factor} and target {target}: ")
            for k,v in result.items():
                print(f"{k}: {v}")
            print("=================================================")