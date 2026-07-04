# Financing Impact Tracking - Metrics Guide

## Overview
This guide explains the metrics added to track how financing costs affect your delta hedging P&L.

---

## Key Metrics Added

### 1. **Financing Cost / Ex-Financing P&L** ⭐ (Your Suggested Metric)
```
= Total Financing Impact / P&L ex-Financing × 100%
```
**Purpose**: Shows financing as a percentage of your "core" strategy performance  
**Interpretation**: 
- Positive % = Financing adds to P&L (e.g., earning cash rate on positive cash)
- Negative % = Financing reduces P&L (e.g., paying borrow rate on short stock)
- Example: -15% means financing costs are eating up 15% of your core strategy profit

---

### 2. **Financing Cost / Total P&L**
```
= Total Financing Impact / P&L with Financing × 100%
```
**Purpose**: Shows financing as a percentage of your final result  
**Interpretation**: Useful when comparing overall strategy returns

---

### 3. **Average Daily Financing** (Historical only)
```
= Total Financing Impact / Number of Trading Days
```
**Purpose**: Shows the daily "carrying cost" of your positions  
**Use case**: Budget planning and daily P&L expectations

---

### 4. **Distribution Analysis** (Monte Carlo only)
Shows financing ratios across different percentiles (mean, median, 5th, 95th)  
**Purpose**: Understand how financing impact varies across different scenarios

---

## Where to Find These Metrics

### Historical Backtest (Tab 1)
- **Section 6**: "P&L Impact of Financing"
- **Visualizations**:
  1. Time series showing P&L with/without financing over time
  2. Bar chart comparing final values
  3. Metrics cards with key ratios
  4. Detailed summary table

### Monte Carlo Simulation (Tab 2)
- **Section**: "Mean P&L: Financing Impact" (after distributions)
- **Visualizations**:
  1. Bar chart showing mean values across all paths
  2. Metrics cards with mean financing ratios
  3. Detailed statistics table (mean, median, std, percentiles)
  4. Financing cost ratios across distribution

---

## Alternative Metrics to Consider

### 1. **Financing Cost per Unit Notional**
```
= Total Financing / (Average Stock Position × Average Stock Price)
```
Shows financing cost relative to the size of your hedge

### 2. **Financing Cost per Option Contract**
```
= Total Financing / Number of Option Contracts
```
Useful for scaling analysis

### 3. **Financing Efficiency Ratio**
```
= P&L ex-Financing / |Total Financing Cost|
```
Shows how much P&L you generate per dollar of financing cost  
Higher is better

### 4. **Return on Capital (ROC) Analysis**
If you track initial capital required:
```
ROC with Financing = Final P&L with Fin / Initial Capital
ROC ex-Financing = Final P&L ex Fin / Initial Capital
Financing Drag = ROC with Fin - ROC ex-Fin
```

### 5. **Break-Even Financing Rate**
Calculate what financing rate would zero out your P&L  
Useful for sensitivity analysis

---

## Recommendations

### For Day-to-Day Monitoring:
1. **Financing Cost / Ex-Fin P&L** - Your best single metric
2. **Average Daily Financing** - Track daily patterns

### For Strategy Evaluation:
1. Compare mean P&L across different borrow/cash rate scenarios
2. Look at financing ratios across percentiles (5th, 50th, 95th)
3. Evaluate if financing cost is stable or varies significantly

### For Risk Management:
1. Monitor maximum daily financing cost (worst case)
2. Track financing as % of total P&L over time
3. Set alerts if financing exceeds certain thresholds (e.g., >20% of P&L)

### For Optimization:
1. Test different hedging frequencies to reduce position sizes → lower financing
2. Analyze cash rate vs borrow rate sensitivity
3. Consider financing impact when setting bands (wider bands = fewer rehedges = lower financing)

---

## Example Interpretation

**Scenario**: SPY short call delta hedge
- Final P&L with Financing: $1,200
- Final P&L ex-Financing: $2,000
- Financing Impact: -$800
- **Financing Cost / Ex-Fin P&L: -40%**

**Interpretation**: 
Your core strategy (selling volatility + delta hedging) would have made $2,000, but financing costs consumed 40% of that profit, leaving you with only $1,200. This suggests:
- Borrowing costs for short stock are significant
- Consider if the strategy is still attractive after financing
- Might want to reduce hedging frequency or adjust bands to minimize stock positions

---

## Next Steps

1. Run backtests with different cash_rate and borrow_rate values
2. Compare financing impact across different market conditions (volatility regimes)
3. Use Monte Carlo sweep to find optimal hedging_vol that balances P&L vs financing
4. Track these metrics over multiple backtests to build intuition

---

**Note**: All these metrics are automatically calculated and displayed in the Streamlit dashboard after running backtests or simulations.

