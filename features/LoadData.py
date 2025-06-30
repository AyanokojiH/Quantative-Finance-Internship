'''
Features to be designed:
1. Load column 21~ from dataset/intern_data/risk_data, indicating each stock's industry;
2. Load VWAP_PRICE_2 from dataset/intern_data/basic_data, meanwhile pay attention to the following signs:
    1) TRADE_DATE_GAP: Not zero means that during the past numeric day the stock wasn't on trade.
    2) HIGHEST_PRICE_2 V.S. LOWEST_PRICE_2: Not zero means that it rised or fell to stop.
3. Calculate 5-day-return rate(t0 v.s. t-5);Calculate earn rate(t1 v.s. t2);
4. Merge all the above information into a bigger dataframe, whose index is TRADE_DATE, under every TRADE_DATE, list every
   stock's 5-day-return rate, as well as its earn rate.
'''


import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from utils import Factor_5D_reverse, Factor_Relative_Strength,Factor_Weighted_Strength_1m
import time

class StockDataProcessor:
    def __init__(self,basic_path, risk_path, output_path,
                 return_short, return_med, return_long, trade_threshold, factor):
        self.basic_path = Path(basic_path)
        self.risk_path = Path(risk_path)
        self.output_path = Path(output_path)
        self.return_short = return_short
        self.return_med = return_med
        self.return_long = return_long
        self.trade_threshold = trade_threshold
        self.factor = factor
        self.merged_data = None

    def _load_single_stock_data(self, stock_code):
        """Load and process data for a single stock"""
        try:
            # Load basic data (TRADE_DATE is index)
            basic_file = self.basic_path / f"{stock_code}.parquet"
            basic_df = pd.read_parquet(basic_file, columns=[
                'VWAP_PRICE_2', 'TRADE_DATE_COUNTER','MARKET_VALUE','TURNOVER_VOL','NEG_MARKET_VALUE','TURNOVER_RATE'
                ,'CLOSE_PRICE_2','HIGHEST_PRICE_2', 'LOWEST_PRICE_2'
            ],engine='pyarrow')
            basic_df = basic_df.rename_axis('TRADE_DATE').reset_index()
            
            # Filter problematic trades
            basic_df = basic_df[
                # (basic_df['TRADE_DATE_GAP'] == 0) &
                (basic_df['HIGHEST_PRICE_2'] - basic_df['LOWEST_PRICE_2'] > self.trade_threshold) &
                (basic_df['TRADE_DATE_COUNTER'] >= 60) &
                (basic_df['TURNOVER_VOL']  > 0)
            ]
            
            # Load risk data (TRADE_DATE is index, columns 21+)
            risk_file = self.risk_path / f"{stock_code}.parquet"
            risk_cols = pq.read_table(risk_file).schema.names[20:]
            risk_df = pd.read_parquet(risk_file, columns=risk_cols)
            risk_df = risk_df.rename_axis('TRADE_DATE').reset_index()
            
            # Merge and calculate returns
            merged = pd.merge(basic_df, risk_df, on='TRADE_DATE', how='left')
            merged['STOCK_CODE'] = stock_code
            
            # Calculate returns
            merged = merged.sort_values('TRADE_DATE')
            merged[f'{self.factor}'] = Factor_5D_reverse.calculate_5d_reverse(merged)

            merged[f'NEXT_{self.return_short}DAY_RETURN_RATIO'] = merged.groupby('STOCK_CODE')['VWAP_PRICE_2'].transform(
                lambda x: (x.shift(-(1+self.return_short)) - x.shift(-1)) / x.shift(-1)
            )
            merged[f'NEXT_{self.return_med}DAY_RETURN_RATIO'] = merged.groupby('STOCK_CODE')['VWAP_PRICE_2'].transform(
                lambda x: (x.shift(-(1+self.return_med)) - x.shift(-1)) / x.shift(-1)
            )
            merged[f'NEXT_{self.return_long}DAY_RETURN_RATIO'] = merged.groupby('STOCK_CODE')['VWAP_PRICE_2'].transform(
                lambda x: (x.shift(-(1+self.return_long)) - x.shift(-1)) / x.shift(-1)
            )
            
            return merged.dropna(subset=[f'{self.factor}', f'NEXT_{self.return_short}DAY_RETURN_RATIO', 
                                       f'NEXT_{self.return_med}DAY_RETURN_RATIO', f'NEXT_{self.return_long}DAY_RETURN_RATIO'])
        
        except Exception as e:
            print(f"Error processing {stock_code}. Info: {e}")
            time.sleep(10)
            return None

    def process_all_stocks(self):
        """Process all stocks in the dataset"""
        all_stocks = sorted([f.stem for f in self.basic_path.glob("*.parquet")])
        processed_dfs = []
        
        for stock_code in tqdm(all_stocks, desc="Processing stocks",mininterval=2.0):
            stock_data = self._load_single_stock_data(stock_code)
            if stock_data is not None:
                processed_dfs.append(stock_data)
        
        # Create MultiIndex DataFrame
        self.merged_data = pd.concat(processed_dfs)
        self.merged_data = self.merged_data.set_index(['STOCK_CODE', 'TRADE_DATE'])
        self.merged_data = self.merged_data.sort_index()
        return self.merged_data

    def save_results(self):
        """Save the processed data"""
        self.output_path.mkdir(exist_ok=True)
        self.merged_data.to_parquet(self.output_path / "all_stocks_processed.parquet")

def execute(basic_path,risk_path,output_path,return_short,return_med,return_long,trade_threshold,factor):
    processor = StockDataProcessor(basic_path,risk_path,output_path,return_short,return_med,return_long,trade_threshold,factor=factor)
    result = processor.process_all_stocks()
    processor.save_results()
    print(f"\nProcessing completed. Final data shape: {result.shape}")
    print("Sample output:")
    print(result.head(10))

if __name__ == "__main__":
    execute()

# Idealistically, after running the code, a 2-layer dataframe like 
"""
          TRADE_DATE
STOCK_CODE ...      ...
           ...      ...
           ...      ...
           ...      ...
"""
# would be in "./dataset/processed_data/all_stocks_processed.parquet"