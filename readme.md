# Quantitative Finance - Single Factor Research Project

## Project Overview
A standardized factor research pipeline implementing the Huatai quant methodology. Performs full-cycle factor analysis from computation to performance evaluation, including:
- Data preprocessing
- Standardization/neutralization
- Return prediction testing
- IC/Rank IC analysis
- Backtesting

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
|   |__ FactorBacktest.py
|   
│
├── main.py              # Pipeline controller
└── environment.yml
```

## Note:

This version:
- Maintains technical accuracy
- Focuses on essential information
- Uses clear section headers
- Avoids unnecessary decorations
- Presents the core functionality you built
