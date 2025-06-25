"""
Features to be designed: Apply neutralization on the factor to avoid been influenced by MARKET_VALUE and industry.
First, Regress y[factor_value] on X[dummies, Log(Marketvalue)], use every residual as the factor's new value
Then, Standardize the new factor's value.
"""


import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class FactorNeutralizer:
    def __init__(self,input_path,output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.data = None
        self.factor_columns = [
            '5D_RETURN.median_std', 
            '5D_RETURN.rank_std'
        ]
    
    def load_data(self):
        """Load standardized data with industry dummies and market cap"""
        self.data = pd.read_parquet(self.input_path)
        
        # Verify columns
        self.industry_cols = [col for col in self.data.columns[5:42]] 
        if not self.industry_cols:
            raise ValueError("No industry columns found")
        if 'MARKET_VALUE' not in self.data.columns:
            raise ValueError("MARKET_VALUE column missing")
        
        # Prepare log market cap
        self.data['log_mcap'] = np.log(self.data['MARKET_VALUE'].astype(float))
        print(f"Loaded data with {len(self.industry_cols)} industries and market cap")

    def industry_mcap_neutralization(self, factor_series):
        """
        Neutralize factor by industry + market cap using regression
        Returns: residual series (neutralized factor)
        """
        # 确保数据是数值类型
        factor_series = pd.to_numeric(factor_series, errors='coerce')
        
        # 准备设计矩阵
        industry_dummies = pd.get_dummies(
            self.data[self.industry_cols].idxmax(axis=1),
            prefix='ind',
            drop_first=True
        )
        
        # 确保所有数据都是数值类型
        X = industry_dummies.astype(float)
        X['log_mcap'] = self.data['log_mcap'].astype(float)
        X = sm.add_constant(X)  # 添加截距
        
        # 对齐数据，删除缺失值
        valid_idx = factor_series.notna() & X.notna().all(axis=1)
        y = factor_series[valid_idx]
        X = X[valid_idx]
        
        # 确保有足够的数据
        if len(y) < 10 or X.shape[1] >= len(y):
            print(f"Warning: Not enough data for regression - {len(y)} samples")
            return pd.Series(np.nan, index=factor_series.index)
            
        # 拟合OLS模型
        model = sm.OLS(y, X).fit()
        residuals = pd.Series(np.nan, index=factor_series.index)
        residuals[valid_idx] = model.resid
        return residuals

    def check_original_r2(self, factor_col):
        """检查原始因子受行业/市值解释的程度"""
        # 准备行业哑变量（排除基准行业）
        industry_dummies = pd.get_dummies(
            self.data[self.industry_cols].idxmax(axis=1),
            prefix='ind',
            drop_first=True
        ).astype(float)
        
        # 准备设计矩阵：行业 + 对数市值
        X = pd.concat([industry_dummies, self.data[['log_mcap']]], axis=1)
        X = sm.add_constant(X)  # 添加截距项
        
        # 提取因子值并对齐有效数据
        y = self.data[factor_col]
        valid_idx = y.notna() & X.notna().all(axis=1)
        
        if valid_idx.sum() < 10:  # 至少需要10个样本
            print(f"⚠️ 数据不足: {factor_col} 仅有 {valid_idx.sum()} 个有效样本")
            return np.nan
        
        # 计算原始R²
        model = sm.OLS(y[valid_idx], X[valid_idx]).fit()
        return model.rsquared

    def process_factors(self):
        """处理所有目标因子"""
        for factor in self.factor_columns:
            if factor not in self.data.columns:
                print(f"⚠️ 跳过: 未找到因子列 {factor}")
                continue
                
            # 先检查原始因子的R²
            original_r2 = self.check_original_r2(factor)
            
            # 执行中性化
            new_col = factor.replace('_std', '_neutral')
            self.data[new_col] = self.industry_mcap_neutralization(self.data[factor])
            
            # 重新标准化中性化后的因子
            if self.data[new_col].notna().sum() > 0:
                self.data[new_col] = (
                    (self.data[new_col] - self.data[new_col].mean()) / 
                    self.data[new_col].std()
                )
            
            # 检查中性化后的R²
            neutralized_r2 = self.check_original_r2(new_col)
            print("━" * 50)

    def check_neutralization(self, original_col, neutralized_col):
        """Check if neutralization worked by comparing R-squared"""
        if neutralized_col not in self.data.columns:
            return np.nan
            
        # Check industry effect
        industry_dummies = pd.get_dummies(
            self.data[self.industry_cols].idxmax(axis=1),
            prefix='ind',
            drop_first=True
        ).astype(float)
        X_ind = sm.add_constant(industry_dummies)
        y = self.data[neutralized_col]
        valid_idx = y.notna() & X_ind.notna().all(axis=1)
        
        if valid_idx.sum() > X_ind.shape[1]:  # 确保样本数大于特征数
            model_ind = sm.OLS(y[valid_idx], X_ind[valid_idx]).fit()
            ind_r2 = model_ind.rsquared
        else:
            ind_r2 = np.nan
            
        # Check market cap effect
        X_mcap = sm.add_constant(self.data[['log_mcap']].astype(float))
        valid_idx = y.notna() & X_mcap.notna().all(axis=1)
        
        if valid_idx.sum() > 1:  # 至少需要2个样本
            model_mcap = sm.OLS(y[valid_idx], X_mcap[valid_idx]).fit()
            mcap_r2 = model_mcap.rsquared
        else:
            mcap_r2 = np.nan
            
        return max(ind_r2, mcap_r2) if not np.isnan([ind_r2, mcap_r2]).all() else np.nan

    def save_results(self):
        """Save neutralized data"""
        self.data.to_parquet(self.output_path)
        print(f"Saved neutralized data to {self.output_path}")

    def run(self):
        """Execute full pipeline"""
        self.load_data()
        self.process_factors()
        self.save_results()

def execute(input_path,output_path):
    neutralizer = FactorNeutralizer(input_path,output_path)
    neutralizer.run()