# Quantitative Factor Research Framework

## Project Overview
A standardized factor research pipeline implementing the Huatai quant methodology. Performs full-cycle factor analysis from computation to performance evaluation, including:
- Data preprocessing
- Standardization/neutralization
- Return prediction testing
- IC/Rank IC analysis

## Core Features
✅ 5D-Reversal factor analysis  
✅ Industry/market cap neutralization  
✅ Automated IC/IR reporting  
✅ Parametric configuration system  

## Project Structure
```
.
├── config/               # Parameter configurations
│   ├── config.yaml
│   └── Config.py
│
├── dataset/              # Data management
│   ├── intern_data_new       # Original Data
│   └── processed_data       # Data cleaning
│
├── features/             # Factor computations
│   ├── FactorNeutralization.py # Industry/cap neutralization
│   ├── FactorStandardize.py
│   ├── FactorIC.py
│   └── FactorReturn.py
|   
│
├── main.py              # Pipeline controller
└── requirements.txt
```

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure data paths in `config/paths.yaml`

3. Run full pipeline:
```bash
python main.py --factor momentum --start 20200101 --end 20221231
```

## Key Parameters
Configure in `config/factor_params.yaml`:
```yaml
momentum:
  windows: [5, 20, 60]    # Lookback periods
  neutralization: 
    - industry            # Neutralization methods
    - market_cap
```

## Outputs
- IC analysis reports (`results/ic_stats.csv`)
- Factor return curves (`plots/return_curves/`)
- Neutralized factor values (`output/factors/`)

## Development Notes
- All financial data should be sorted by TRADE_DATE
- Uses WINSORIZE(0.05) for outlier handling
- Turnover-based suspension filtering implemented

---

This version:
- Maintains technical accuracy
- Focuses on essential information
- Uses clear section headers
- Avoids unnecessary decorations
- Presents the core functionality you built
