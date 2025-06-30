"""
Factor IC Calculator - Fixed Version
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Tuple, List
import warnings
from datetime import datetime

class FactorICCalculator:
    def __init__(self, data: pd.DataFrame, short: int, med: int, long: int, factor: str):
        self.original_data = data.copy()
        self.data = data.copy()
        self.short = short
        self.med = med
        self.long = long
        self.factor = factor
        
        # Validate required columns
        required_cols = [
            factor,
            f'NEXT_{short}DAY_RETURN_RATIO',
            f'NEXT_{med}DAY_RETURN_RATIO', 
            f'NEXT_{long}DAY_RETURN_RATIO'
        ]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        self._clean_data()

    def _clean_data(self):
        """Remove rows with missing values"""
        self.data = self.original_data.dropna(subset=[
            self.factor,
            f'NEXT_{self.short}DAY_RETURN_RATIO',
            f'NEXT_{self.med}DAY_RETURN_RATIO',
            f'NEXT_{self.long}DAY_RETURN_RATIO'
        ]).copy()

    def reset_data(self):
        self.data = self.original_data.copy()
        self._clean_data()

    def filter_by_year(self, year: int):
        """Filter data by year (0 means all years)"""
        self.reset_data()
        if year == 0:
            return
            
        if 'TRADE_DATE' in self.data.index.names:
            dates = self.data.index.get_level_values('TRADE_DATE')
            self.data = self.data[dates.year == year]
        elif 'TRADE_DATE' in self.data.columns:
            self.data = self.data[pd.to_datetime(self.data['TRADE_DATE']).dt.year == year]
        else:
            raise ValueError("TRADE_DATE not found in index or columns")

    def calculate_ic(self, factor_col: str, return_col: str) -> Dict[str, float]:
        """
        Calculate daily cross-sectional IC and Rank IC
        Returns dict with both scalar metrics and full IC series
        """
        # Get dates for grouping
        if 'TRADE_DATE' in self.data.index.names:
            dates = self.data.index.get_level_values('TRADE_DATE')
        elif 'TRADE_DATE' in self.data.columns:
            dates = pd.to_datetime(self.data['TRADE_DATE'])
        else:
            raise ValueError("TRADE_DATE not found")
        
        # Calculate daily IC and Rank IC
        ic_list, rank_ic_list = [], []
        date_list = []
        
        for date, group in self.data.groupby(dates):
            if len(group) < 2:  # Need at least 2 stocks to calculate correlation
                continue
                
            # Pearson IC
            with warnings.catch_warnings():  # Ignore warnings for constant values
                warnings.simplefilter("ignore")
                ic = group[factor_col].corr(group[return_col])
            
            # Spearman Rank IC
            rank_ic, _ = spearmanr(group[factor_col], group[return_col])
            
            if not np.isnan(ic):
                ic_list.append(ic)
                rank_ic_list.append(rank_ic)
                date_list.append(date)
        
        # Create time series
        ic_series = pd.Series(ic_list, index=date_list, name='IC')
        rank_ic_series = pd.Series(rank_ic_list, index=date_list, name='RankIC')
        
        # Calculate statistics
        valid_ics = ic_series.dropna()
        valid_rank_ics = rank_ic_series.dropna()
        
        return {
            'factor': factor_col,
            'target': return_col,
            'IC': valid_ics.mean(),
            'IC_std': valid_ics.std(),
            'AbsIC': np.abs(valid_ics).mean(),
            'RankIC': valid_rank_ics.mean(),
            'RankIC_std': valid_rank_ics.std(),
            'AbsRankIC': np.abs(valid_rank_ics).mean(),
            'IC_series': ic_series,
            'RankIC_series': rank_ic_series,
            'n_obs': len(valid_ics)
        }

    def calculate_icir(self, factor_col: str) -> Dict[str, float]:
        """
        Calculate annual ICIR using daily IC series
        ICIR = |mean(daily_IC)| / std(daily_IC)
        """
        # Get daily IC series
        ic_result = self.calculate_ic(factor_col, f'NEXT_{self.short}DAY_RETURN_RATIO')
        ic_series = ic_result['IC_series']
        
        if len(ic_series) == 0:
            return {
                'ICIR': np.nan,
                'IC_mean': np.nan,
                'IC_std': np.nan,
                'RankIC_mean': np.nan,
                'RankIC_std': np.nan,
                'n_years': 0
            }
        
        # Calculate full-period ICIR (all years)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        
        # Count valid years (years with at least 10 trading days)
        n_years = sum(ic_series.groupby(ic_series.index.year).count() >= 10)
        
        return {
            'ICIR': np.abs(ic_mean) / ic_std if ic_std != 0 else np.nan,
            'IC_mean': ic_mean,
            'IC_std': ic_std,
            'RankIC_mean': np.nan,  # Would need to calculate separately
            'RankIC_std': np.nan,
            'n_years': n_years
        }

    def calculate_all_metrics(self, factor_col: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Calculate IC for all return periods and ICIR"""
        targets = [
            f'NEXT_{self.short}DAY_RETURN_RATIO',
            f'NEXT_{self.med}DAY_RETURN_RATIO',
            f'NEXT_{self.long}DAY_RETURN_RATIO'
        ]
        
        ic_results = []
        for target in targets:
            try:
                ic_result = self.calculate_ic(factor_col, target)
                ic_results.append(ic_result)
            except Exception as e:
                print(f"Error calculating {target}: {str(e)}")
                continue
        
        # Convert to DataFrame (keep only scalar metrics)
        results_df = pd.DataFrame([
            {k: v for k, v in res.items() if not isinstance(v, pd.Series)}
            for res in ic_results
        ])
        
        # Calculate ICIR (only for short-term return)
        icir_info = self.calculate_icir(factor_col)
        
        return results_df, icir_info

def print_results(results_df: pd.DataFrame, icir_info: Dict[str, float], year: str):
    """Save results to file"""
    def format_float(x):
        return f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}"
    
    with open('./features/IC_Info.txt', 'a') as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"RESULTS FOR {year}\n")
        f.write("="*60 + "\n")
        
        f.write(f"\nICIR: {format_float(icir_info.get('ICIR', np.nan))}\n")
        f.write(f"Mean IC: {format_float(icir_info.get('IC_mean', np.nan))}\n")
        f.write(f"IC Std: {format_float(icir_info.get('IC_std', np.nan))}\n")
        f.write(f"Valid Years: {icir_info.get('n_years', 0)}\n")
        
        # Prepare table columns
        columns = [
            ('Target', 'target', lambda x: x.replace('_RETURN_RATIO', '')),
            ('IC', 'IC', format_float),
            ('AbsIC', 'AbsIC', format_float),
            ('RankIC', 'RankIC', format_float),
            ('AbsRankIC', 'AbsRankIC', format_float),
            ('Obs', 'n_obs', lambda x: f"{x:,}")
        ]
        
        # Write header
        f.write("\n" + "|".join([f" {col[0]:<10}" for col in columns]) + "|\n")
        f.write("-"*(11*len(columns)+1) + "\n")
        
        # Write data rows
        for _, row in results_df.iterrows():
            f.write("|".join([f" {col[2](row[col[1]]):<10}" for col in columns]) + "|\n")
        
        f.write(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def execute(use_factor: str, short: int, med: int, long: int, factor: str):
    """Main execution function"""
    # Load data
    df = pd.read_parquet("./dataset/processed_data/Neutralization.parquet")
    
    # Initialize calculator
    calculator = FactorICCalculator(df, short, med, long, factor)
    
    # Get available years
    if 'TRADE_DATE' in df.index.names:
        years = sorted(df.index.get_level_values('TRADE_DATE').year.unique())
    else:
        years = sorted(pd.to_datetime(df['TRADE_DATE']).dt.year.unique())
    
    # Add 'All Years' option (0)
    years = [0] + years
    
    # Process each year
    for year in years:
        calculator.filter_by_year(year)
        ic_results, icir_info = calculator.calculate_all_metrics(factor)
        year_label = "ALL YEARS" if year == 0 else str(year)
        print_results(ic_results, icir_info, year_label)
    
    print("Analysis complete. Results saved to ./features/IC_Info.txt")