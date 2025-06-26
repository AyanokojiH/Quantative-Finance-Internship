from features import FactorIC, FactorNeutralization, FactorReturn, FactorStandardize, FactorBacktest,LoadData
from config import Config
import time

class Pipe:
    def __init__(self):
        self.para = None
        self.status = 0
        self.timings = {} 

    def _time_it(self, func, step_name):
        start_time = time.time()
        func()
        elapsed_time = time.time() - start_time
        self.timings[step_name] = elapsed_time
        return elapsed_time

    def run(self):
        # 步骤0：读取参数
        self.para = Config.run_config()
        print("Parameters Loaded Successfully. Waiting...")
        
        # 步骤1: 加载数据
        print("\nCurrent Process: LoadData(1/6). In this step we load data from the dataset")
        print("And calculate the values of the factors.")
        load_time = self._time_it(
            lambda: LoadData.execute(
                basic_path=self.para['path']['load_data']['basic_data'],
                risk_path=self.para['path']['load_data']['risk_data'],
                output_path=self.para['path']['load_data']['output_data'],
                return_short=self.para['parameters']['return_short'],
                return_med=self.para['parameters']['return_med'],
                return_long=self.para['parameters']['return_long'],
                trade_threshold=self.para['parameters']['trade_threshold']
            ),
            "LoadData"
        )
        print(f"-> Completed in {load_time:.2f} seconds")
        print("="*50, end='\n\n')
        
        # 步骤2: 因子标准化
        print("Current Process: FactorStandardize(2/6). In this step we standardize the factor's value")
        print("so as to avoid influence from extreme values.")
        std_time = self._time_it(
            lambda:FactorStandardize.execute(
                input_path = self.para['path']['Factor_Standardize']['input_path'],
                output_path = self.para['path']['Factor_Standardize']['output_path'],
                n = self.para['parameters']['standardized_n']
            ), 
            "FactorStandardize")
        print(f"-> Completed in {std_time:.2f} seconds")
        print("="*50, end='\n\n')
        
        # 步骤3: 因子中性化
        print("Current Process: FactorNeutralization(3/6). In this step we neutralize the factor's value")
        print("so as to avoid influence from different industries and MarketValues.")
        neutral_time = self._time_it(lambda: FactorNeutralization.execute(
            input_path = self.para['path']['Factor_Neutralization']['input_path'],
            output_path = self.para['path']['Factor_Neutralization']['output_path'],
        ), 
        "FactorNeutralization")
        print(f"-> Completed in {neutral_time:.2f} seconds")
        print("="*50, end='\n\n')
        
        # 步骤4: 计算因子收益
        print("Current Process: FactorReturn_Calculating(4/6). In this step we calculate all the factor's return.")
        print("A brief report will be given.")
        return_time = self._time_it(lambda:FactorReturn.execute(
            input_path= self.para['path']['Factor_Return']['read_path'],
            short=self.para['parameters']['return_short'],
            med=self.para['parameters']['return_med'],
            long=self.para['parameters']['return_long'],
            backtest = self.para['path']['Factor_Return']['backtest']
        ), "FactorReturn")
        print(f"-> Completed in {return_time:.2f} seconds")
        print("="*50, end='\n\n')
        
        # 步骤5: 计算IC值
        print("Current Process: FactorIC_Calculating(5/6). In this step we calculate all the factor's IC,Rank-IC and ICIR.")
        print("A brief report will be given.")
        ic_time = self._time_it(lambda: FactorIC.execute(
            use_factor= self.para['path']['Factor_IC']['use_factor'],
            short=self.para['parameters']['return_short'],
            med=self.para['parameters']['return_med'],
            long=self.para['parameters']['return_long']
        ), "FactorIC")
        print(f"-> Completed in {ic_time:.2f} seconds")
        print("="*50, end='\n\n')

        # 步骤6：因子回测
        print("Current Process: FactorBacktesting(6/6). In this step we will run a decile portfolio backtest to see the factor's performance.")
        print("Too see the details, visit ./dataset/processed_data/backtest_results.")
        backtest_time = self._time_it(lambda: FactorBacktest.execute(
            input_path=self.para['path']['Factor_Backtest']['input_path'],
            output_path= self.para['path']['Factor_Backtest']['output_dir'],
            short=self.para['parameters']['return_short'],
            med=self.para['parameters']['return_med'],
            long=self.para['parameters']['return_long']
        ),"FactorBacktest")        
        self._print_timing_summary()

    def _print_timing_summary(self):
        print("\n" + "="*50)
        print("Process Timing Summary:")
        print("-"*50)
        for step, duration in self.timings.items():
            print(f"{step:<20}: {duration:.2f} seconds")
        
        total_time = sum(self.timings.values())
        print("-"*50)
        print(f"{'TOTAL TIME':<20}: {total_time:.2f} seconds")
        print("="*50)
        print("\nAll processes have been finished successfully.")

if __name__ == "__main__":
    pipe = Pipe()
    pipe.run()