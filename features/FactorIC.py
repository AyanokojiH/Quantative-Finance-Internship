"""
Features to be designed:
Calculate the IC and Rank-IC value for the factor.
    - IC = Corr(Factor\t, Return\t+1)
    - Rank-IC = SpearmanCorr(rank(Factor\t),rank(Return\t+1))
    - ICIR = mean(IC)/std(IC)
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple

class FactorICCalculator:
    def __init__(self, data: pd.DataFrame, short, med, long):
        self.original_data = data.copy()  
        self.data = data.copy()
        self.required_columns = ['5D_RETURN.rank_std', f'NEXT_{short}DAY_RETURN_RATIO',
                               f'NEXT_{med}DAY_RETURN_RATIO', f'NEXT_{long}DAY_RETURN_RATIO']
        self._validate_data()
        self._clean_data()
        self.short ,self.med, self.long = short,med,long

    def _validate_data(self) -> None:
        missing_cols = [col for col in self.required_columns if col not in self.original_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _clean_data(self) -> None:
        self.data = self.original_data.dropna(subset=self.required_columns).copy()

    def reset_data(self) -> None:
        self.data = self.original_data.dropna(subset=self.required_columns).copy()

    def filter_by_year(self, year: int) -> None:
        self.reset_data()
        if year == 0:
            return
        
        if 'TRADE_DATE' in self.data.index.names:
            self.data = self.data[self.data.index.get_level_values('TRADE_DATE').year == year]
        else:
            self.data = self.data[pd.to_datetime(self.data['TRADE_DATE']).dt.year == year]

    def calculate_ic(self, factor_col: str, return_col: str) -> Dict[str, float]:
        cov = np.cov(self.data[factor_col], self.data[return_col])[0, 1]
        
        factor_std = self.data[factor_col].std()
        return_std = self.data[return_col].std()
        
        ic = cov / (factor_std * return_std)
        
        factor_z = (self.data[factor_col] - self.data[factor_col].mean()) / factor_std
        return_z = (self.data[return_col] - self.data[return_col].mean()) / return_std
        pointwise_ic = factor_z * return_z
        
        abs_ic = np.mean(np.abs(ic))

        rank_ic, rank_p = spearmanr(self.data[factor_col], self.data[return_col])
        
        factor_rank = self.data[factor_col].rank()
        return_rank = self.data[return_col].rank()
        factor_rank_z = (factor_rank - factor_rank.mean()) / factor_rank.std()
        return_rank_z = (return_rank - return_rank.mean()) / return_rank.std()
        pointwise_rank_ic = factor_rank_z * return_rank_z
        
        abs_rank_ic = np.mean(np.abs(rank_ic))
        
        return {
            'factor': factor_col,
            'target': return_col,
            'IC': ic,
            'AbsIC': abs_ic,
            'RankIC': rank_ic,
            'AbsRankIC': abs_rank_ic,
            'RankIC_pvalue': rank_p,
            'n_obs': len(self.data)
        }

    def calculate_icir(self, factor_col: str, window: int = 12) -> Dict[str, float]:
        """修正后的ICIR计算方法（处理空数据情况）"""
        if 'TRADE_DATE' not in self.data.columns and 'TRADE_DATE' not in self.data.index.names:
            raise ValueError("TRADE_DATE column required for ICIR calculation")
            
        # 准备日期数据
        if 'TRADE_DATE' in self.data.index.names:
            dates = pd.to_datetime(self.data.index.get_level_values('TRADE_DATE'))
        else:
            dates = pd.to_datetime(self.data['TRADE_DATE'])
        
        # 按月分组
        self.data['MONTH'] = dates.to_period('M')
        months = self.data['MONTH'].unique()
        
        monthly_ics = []
        monthly_rank_ics = []
        
        for month in months:
            monthly_data = self.data[self.data['MONTH'] == month]
            if len(monthly_data) < 10:  # 最小样本量要求
                continue
                
            # 计算月度IC和RankIC
            ic = monthly_data[factor_col].corr(monthly_data[f'NEXT_{self.short}DAY_RETURN_RATIO'])
            rank_ic, _ = spearmanr(monthly_data[factor_col], monthly_data[f'NEXT_{self.short}DAY_RETURN_RATIO'])
            
            monthly_ics.append(ic)
            monthly_rank_ics.append(rank_ic)
        
        # 处理数据不足的情况
        if len(monthly_ics) == 0:
            return {
                'ICIR': np.nan,
                'IC_mean': np.nan,
                'IC_std': np.nan,
                'RankIC_mean': np.nan,
                'RankIC_std': np.nan,
                'n_months': 0
            }
        
        # 转换为Series便于计算滚动统计量
        ic_series = pd.Series(monthly_ics)
        rank_ic_series = pd.Series(monthly_rank_ics)
        
        # 计算滚动统计量（仅当数据足够时）
        result = {
            'n_months': len(monthly_ics)
        }
        
        # IC相关指标
        if len(ic_series) >= window:
            rolling_ic = ic_series.rolling(window)
            result.update({
                'ICIR': rolling_ic.mean().iloc[-1] / rolling_ic.std().iloc[-1] if rolling_ic.std().iloc[-1] != 0 else np.nan,
                'IC_mean': rolling_ic.mean().iloc[-1],
                'IC_std': rolling_ic.std().iloc[-1]
            })
        else:
            result.update({
                'ICIR': np.nan,
                'IC_mean': np.nan,
                'IC_std': np.nan
            })
        
        # RankIC相关指标
        if len(rank_ic_series) >= window:
            rolling_rank_ic = rank_ic_series.rolling(window)
            result.update({
                'RankIC_mean': rolling_rank_ic.mean().iloc[-1],
                'RankIC_std': rolling_rank_ic.std().iloc[-1]
            })
        else:
            result.update({
                'RankIC_mean': np.nan,
                'RankIC_std': np.nan
            })
        
        return result

    def calculate_all_metrics(self, factor_col: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Calculate both IC values and ICIR"""
        targets = [f'NEXT_{self.short}DAY_RETURN_RATIO', f'NEXT_{self.med}DAY_RETURN_RATIO', f'NEXT_{self.long}DAY_RETURN_RATIO']
        ic_results = pd.DataFrame([self.calculate_ic(factor_col, target) for target in targets])
        icir_info = self.calculate_icir(factor_col)
        return ic_results, icir_info

def format_float(x: float) -> str:
    return f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}"

def print_results(results_df: pd.DataFrame, icir_info: Dict[str, float], year: str) -> None:
    with open('./features/IC_Info.txt', 'a') as f:

        f.write("\n" + "="*60 + "\n")
        f.write(f"RESULTS FOR {year}\n")
        f.write("="*60 + "\n")
        
        f.write(f"\nICIR (12-month rolling): {format_float(icir_info.get('ICIR', np.nan))}\n")
        f.write(f"Mean IC: {format_float(icir_info.get('IC_mean', np.nan))}\n")
        f.write(f"IC Std: {format_float(icir_info.get('IC_std', np.nan))}\n")
        f.write(f"Months included: {icir_info.get('n_months', 0)}\n")
        
        output_columns = [
            ('Target Return', 'target', lambda x: x.replace('_RETURN_RATIO', '')),
            ('IC', 'IC', format_float),
            ('AbsIC', 'AbsIC', format_float),
            ('RankIC', 'RankIC', format_float),
            ('AbsRankIC', 'AbsRankIC', format_float),
            ('RankIC p-value', 'RankIC_pvalue', lambda x: f"{x:.4f}"),
            ('Obs Count', 'n_obs', lambda x: f"{x:,}")
        ]
        
        f.write("\n" + "|".join([f" {col[0]:<15}" for col in output_columns]) + "|\n")
        f.write("-"*(16*len(output_columns)+1) + "\n")
        
        for _, row in results_df.iterrows():
            f.write("|".join([f" {col[2](row[col[1]]):<15}" for col in output_columns]) + "|\n")

        from datetime import datetime
        f.write(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# def generate_summary_table(calculator: FactorICCalculator, years: List[int], factor_col: str, short, med, long) -> pd.DataFrame:
#     results = []
#     for year in years:
#         if year == 0:
#             continue  # Skip "ALL YEARS"
            
#         calculator.filter_by_year(year)
        
#         # Initialize dictionary to store yearly results
#         year_result = {'Year': year}
        
#         # Calculate metrics for each target return period
#         for target in [f'NEXT_{short}DAY_RETURN_RATIO', f'NEXT_{med}DAY_RETURN_RATIO', f'NEXT_{long}DAY_RETURN_RATIO']:
#             # Calculate metrics
#             metrics = calculator.calculate_ic(factor_col, target)
            
#             # Store the metrics with appropriate prefixes
#             prefix = target.replace('_RETURN_RATIO', '')
#             year_result.update({
#                 f'{prefix}_IC': metrics['IC'],
#                 f'{prefix}_AbsIC': metrics['AbsIC'],
#                 f'{prefix}_RankIC': metrics['RankIC'],
#                 f'{prefix}_AbsRankIC': metrics['AbsRankIC'],
#             })
            
#             # Calculate ICIR for each target (using monthly data)
#             if 'TRADE_DATE' in calculator.data.index.names:
#                 dates = pd.to_datetime(calculator.data.index.get_level_values('TRADE_DATE'))
#             else:
#                 dates = pd.to_datetime(calculator.data['TRADE_DATE'])
            
#             monthly_ics = []
#             for month in dates.to_period('M').unique():
#                 month_data = calculator.data[dates.to_period('M') == month]
#                 if len(month_data) < 10:
#                     continue
#                 ic = month_data[factor_col].corr(month_data[target])
#                 monthly_ics.append(ic)
            
#             if len(monthly_ics) >= 12:
#                 year_result[f'{prefix}_ICIR'] = np.mean(monthly_ics) / np.std(monthly_ics) if np.std(monthly_ics) != 0 else np.nan
#             else:
#                 year_result[f'{prefix}_ICIR'] = np.nan
        
#         results.append(year_result)
    
#     # Create DataFrame with ordered columns
#     summary_df = pd.DataFrame(results)
    
#     # Define column order
#     columns_order = ['Year']
#     for prefix in [f'NEXT_{short}DAY', f'NEXT_{med}DAY', f'NEXT_{long}DAY']:
#         columns_order.extend([
#             f'{prefix}_IC',
#             f'{prefix}_AbsIC',
#             f'{prefix}_RankIC',
#             f'{prefix}_AbsRankIC',
#             f'{prefix}_ICIR'
#         ])
    
#     return summary_df[columns_order].set_index('Year')

def execute(use_factor, short, med, long):
    # Load data
    df = pd.read_parquet("./dataset/processed_data/Neutralization.parquet")
    calculator = FactorICCalculator(df,short, med, long)
    
    # Get available years
    if 'TRADE_DATE' in df.index.names:
        years = sorted(df.index.get_level_values('TRADE_DATE').year.unique())
    else:
        years = sorted(pd.to_datetime(df['TRADE_DATE']).dt.year.unique())
    years = years  # 0 represents all years
    
    factor = use_factor
    
    # Generate detailed annual reports
    for year in years:
        calculator.filter_by_year(year)
        ic_results, icir_info = calculator.calculate_all_metrics(factor)
        year_label = "ALL YEARS" if year == 0 else str(year)
        print_results(ic_results, icir_info, year_label)
    
    print("See details in ./docs/result_IC.txt.")
    
    # Generate summary table
    # summary_table = generate_summary_table(calculator, years[1:], factor, short, med, long)
    
    # # Format display
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 1000)
    # pd.set_option('display.float_format', '{:.4f}'.format)
    
    # print("\n" + "="*120)
    # print("Five-Day Reversal Factor Performance Detailed Metrics (2010-2022)")
    # print("="*120)
    # print(summary_table)