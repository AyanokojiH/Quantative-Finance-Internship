"""
Features to be designed:

Backtest the factor in decile based on the stock's short, medium and long term return rate.
We expected the factor to be effective, which means that Decile 10 must significantly lose to Decile 1.
Also, the market can't be determinative to the return rate.

"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class EnhancedBacktester:
    def __init__(self, data_path, output_dir,short,med, long):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data = None
        self.factors = ['5D_RETURN.median_neutral', '5D_RETURN.rank_neutral']
        self.ret_cols = [f'NEXT_{short}DAY_RETURN_RATIO', f'NEXT_{med}DAY_RETURN_RATIO', f'NEXT_{long}DAY_RETURN_RATIO']

    def load_data(self):
        self.data = pd.read_parquet(self.data_path)
        if not isinstance(self.data.index, pd.MultiIndex):
            raise ValueError("Error in input data's form.")
        
        self.data['log_mcap'] = np.log(self.data['NEG_MARKET_VALUE'])
        print(f"Loading finished. Timing range : {self.data.index.get_level_values('TRADE_DATE').min()} 至 "
              f"{self.data.index.get_level_values('TRADE_DATE').max()}")

    def create_decile_portfolios(self, factor_col):
        if factor_col not in self.data.columns:
            raise ValueError(f"Missing factor:  {factor_col} ")
            
        self.data['decile'] = self.data.groupby('TRADE_DATE', group_keys=False)[factor_col].apply(
            lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop') + 1
        )
        
        return self.data.groupby(['TRADE_DATE', 'decile']).agg({
            'log_mcap': 'mean',
            **{col: 'mean' for col in self.ret_cols}
        }).unstack()

    def calculate_cumulative_mcap(self, portfolio):
        mcap = portfolio['log_mcap']
        cum_mcap = (mcap - mcap.iloc[0]) * 100  
        return cum_mcap

    def plot_cumulative_results(self, cum_mcap, cum_returns, factor_name):
        plt.figure(figsize=(14, 10))
        
        # 1. 累积市值变化 (子图位置1)
        plt.subplot(2, 2, 1)
        for decile in range(1, 11):
            plt.plot(cum_mcap[decile], label=f'Decile {decile}')
        plt.title(f'Cumulative Log Market Cap Change - {factor_name}')
        plt.ylabel('Cumulative % Change (log scale)')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        
        # 2. 累积收益率 (子图位置2-4)
        for i, col in enumerate(self.ret_cols, start=2):
            plt.subplot(2, 2, i)
            for decile in range(1, 11):
                plt.plot(cum_returns[col][decile], label=f'Decile {decile}')
            plt.title(col.replace('_', ' '))
            plt.ylabel('Cumulative Return')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir/f'cumulative_{factor_name}.png', dpi=300)
        plt.close()

    def run_analysis(self):
        self.load_data()
        
        for factor in self.factors:
            try:
                print(f"\nProcessing {factor}...")
                
                portfolio = self.create_decile_portfolios(factor)
                cum_mcap = self.calculate_cumulative_mcap(portfolio)
                cum_returns = portfolio[self.ret_cols].cumsum()
                
                self.plot_cumulative_results(cum_mcap, cum_returns, factor)
            
                cum_mcap.to_csv(self.output_dir/f'cum_mcap_{factor}.csv')
                cum_returns.to_csv(self.output_dir/f'cum_returns_{factor}.csv')
                
                print(f"Successfully processed {factor}")
                
            except Exception as e:
                print(f"Error processing {factor}: {str(e)}")
                continue

        print("\nAnalysis completed. Results saved to:", self.output_dir)

def execute(input_path, output_path, short, med, long):
    backtester = EnhancedBacktester(
        data_path= Path(input_path),
        output_dir=Path(output_path),
        short=short,
        med = med,
        long= long
    )
    backtester.run_analysis()