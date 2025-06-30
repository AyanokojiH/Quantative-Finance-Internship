import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=FutureWarning)

class RobustFactorNeutralizer:
    def __init__(self, input_path: str, output_path: str, factor: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.factor = factor
        self.data = None
        self.industry_cols = []
        self.factor_columns = [
            f'{factor}.median_std', 
            f'{factor}.rank_std'
        ]
        self._type_checks = {
            'MARKET_VALUE': np.float64,
            'log_mcap': np.float64
        }

    def _validate_data_types(self):
        """确保关键列的数据类型正确"""
        type_errors = []
        for col, dtype in self._type_checks.items():
            if col in self.data.columns:
                try:
                    self.data[col] = self.data[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    type_errors.append(f"{col}: {str(e)}")
        
        if type_errors:
            raise TypeError(f"Data type conversion failed:\n" + "\n".join(type_errors))

    def load_data(self):
        """加载并验证数据"""
        self.data = pd.read_parquet(self.input_path)
        
        # 识别行业列
        industry_prefixes = ['Agriculture', 'Automobiles', 'Banks', 'BuildMater', 'Chemicals', 
        'Commerce', 'Computers', 'Conglomerates', 'ConstrDecor', 'Defense',
        'ElectricalEquip', 'Electronics', 'FoodBeverages', 'HealthCare',
        'HomeAppliances', 'Leisure', 'LightIndustry', 'MachineEquip', 'Media',
        'Mining', 'NonbankFinan', 'NonferrousMetals', 'RealEstate', 'Steel',
        'Telecoms', 'TextileGarment', 'Transportation', 'Utilities',
        'BasicChemicals', 'BeautyCare', 'Coal', 'EnvironProtect', 'Petroleum',
        'PowerEquip', 'RetailTrade', 'SocialServices', 'TextileApparel']  
        # Verify columns
        self.industry_cols = [col for col in self.data.columns if any(col.startswith(prefix) for prefix in industry_prefixes)]
        
        if not self.industry_cols:
            raise ValueError("未找到行业分类列")
        
        # 准备市值数据
        if 'MARKET_VALUE' not in self.data.columns:
            raise ValueError("缺少MARKET_VALUE列")
        
        self.data['log_mcap'] = np.log(self.data['MARKET_VALUE'].astype(np.float64))
        self._validate_data_types()
        
        print(f"Loaded data with {len(self.industry_cols)} industried successfully.")

    def _safe_daily_regression(self, y: pd.Series, X: pd.DataFrame) -> Optional[np.ndarray]:
        """安全的每日回归计算"""
        try:
            # 强制类型转换
            X = X.astype(np.float64)
            y = y.astype(np.float64)
            
            # 检查有效样本量
            if len(y) < 10 or X.shape[1] >= len(y):
                return None
                
            # 添加截距项
            X = sm.add_constant(X, has_constant='raise')
            
            # 矩阵秩检查
            if np.linalg.matrix_rank(X) < X.shape[1]:
                return None
                
            model = sm.OLS(y, X).fit()
            return model.resid
            
        except Exception as e:
            print(f"Failed to regress: {str(e)}")
            return None

    def daily_cross_sectional_neutralize(self, factor_series: pd.Series) -> pd.Series:
        """健壮的日频横截面中性化"""
        # 获取日期信息
        if 'TRADE_DATE' in self.data.index.names:
            dates = self.data.index.get_level_values('TRADE_DATE')
        elif 'TRADE_DATE' in self.data.columns:
            dates = pd.to_datetime(self.data['TRADE_DATE'])
        else:
            raise ValueError("No info of trade_date is involved.")
        
        neutralized = pd.Series(np.nan, index=factor_series.index, name=factor_series.name)
        
        for date, daily_data in self.data.groupby(dates):
            try:
                # 准备行业哑变量
                industries = daily_data[self.industry_cols].idxmax(axis=1)
                industry_dummies = pd.get_dummies(industries, drop_first=True)
                
                # 构建设计矩阵
                X = pd.concat([
                    industry_dummies.astype(np.float64),
                    daily_data['log_mcap'].astype(np.float64)
                ], axis=1)
                
                # 对齐因子值
                y = factor_series.loc[daily_data.index]
                valid_idx = y.notna() & X.notna().all(axis=1)
                
                # 执行安全回归
                residuals = self._safe_daily_regression(
                    y=y[valid_idx],
                    X=X[valid_idx]
                )
                
                if residuals is not None:
                    neutralized.loc[daily_data.index[valid_idx]] = residuals
                    
            except Exception as e:
                print(f"An error occured when dealing with {date} : {str(e)}")
                continue
                
        return neutralized

    def process_all_factors(self):
        """处理所有因子列"""
        for factor_col in self.factor_columns:
            if factor_col not in self.data.columns:
                print(f"⚠️ Missing factor: {factor_col}")
                continue
                
            print(f"\nDeal with factor: {factor_col}")
            
            # 中性化处理
            neutralized_col = f"{factor_col}_neutral"
            self.data[neutralized_col] = self.daily_cross_sectional_neutralize(
                self.data[factor_col]
            )
            
            # 标准化
            self.data[neutralized_col] = (
                (self.data[neutralized_col] - self.data[neutralized_col].mean()) 
                / self.data[neutralized_col].std()
            )
            
            # 验证结果
            valid_pct = self.data[neutralized_col].notna().mean() * 100
            print(f"Success with accurancy: {valid_pct:.1f}%")

    def save_results(self):
        """保存结果"""
        self.data.to_parquet(self.output_path)
        print(f"\nSaved result to -> {self.output_path}")

    def run(self):
        """执行完整流程"""
        try:
            self.load_data()
            self.process_all_factors()
            self.save_results()
        except Exception as e:
            print(f"Error: {str(e)}")
            raise

def execute(input_path: str, output_path: str, factor: str):
    """执行中性化流程"""
    print(f"\nStarting neutralization with factor: {factor}")
    try:
        neutralizer = RobustFactorNeutralizer(input_path, output_path, factor)
        neutralizer.run()
        print("Neutralization finished. Continuing...")
    except Exception as e:
        print(f"Error occured: {str(e)}")
        raise