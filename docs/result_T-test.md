# Five-Day Reversal Factor Analysis Results

## Key Findings
- **All factors show statistically significant negative returns** (p=0.0), confirming reversal effects
- **Median-neutralized factor slightly outperforms** rank-neutralized version (higher R²)
- **Anomalous decay pattern**: R² peaks at 5-day horizon then declines at 20-day

## Detailed Results

### Factor: `5D_RETURN.rank_neutral`
| Target Return        | Return (%) | t-value    | p-value | R² (%) | Observations |
|-----------------------|------------|------------|---------|--------|--------------|
| NEXT_DAY_RETURN_RATIO | -0.050     | -51.77     | 0.0     | 0.029  | 9,093,094    |
| NEXT_5DAY_RETURN_RATIO| -0.127     | -52.36     | 0.0     | 0.030  | 9,093,094    |
| NEXT_20DAY_RETURN_RATIO| -0.199     | -40.23     | 0.0     | 0.018  | 9,093,094    |

### Factor: `5D_RETURN.median_neutral`
| Target Return        | Return (%) | t-value    | p-value | R² (%) | Observations |
|-----------------------|------------|------------|---------|--------|--------------|
| NEXT_DAY_RETURN_RATIO | -0.053     | -54.75     | 0.0     | 0.033  | 9,093,094    |
| NEXT_5DAY_RETURN_RATIO| -0.137     | -56.49     | 0.0     | 0.035  | 9,093,094    |
| NEXT_20DAY_RETURN_RATIO| -0.224     | -45.32     | 0.0     | 0.023  | 9,093,094    |

## Interpretation

### 1. Statistical Significance
- Extreme t-values (|t| > 40) indicate overwhelming statistical significance
- All p-values effectively zero (p=0.0)

### 2. Economic Significance
- **Daily reversal effect**: ~0.05% return per day
- **Cumulative effects**:
  - 5-day: ~0.13% (non-linear compounding)
  - 20-day: ~0.20-0.22%

### 3. Explanatory Power (R²)
- **Range**: 0.018%-0.035%
- **Pattern**:
    Highest peek appeared at day 5, which suggested that the factor has the strongest explanary validation on the 5th day.