# Delta Hedging Backtest

Interactive dashboard and simulation toolkit for delta-hedged option portfolios. Run historical backtests on real market data, Monte Carlo path simulations, and efficient-frontier analysis to compare hedging strategies.

## Features

### Historical Backtest
- Fetches equity and option-implied data via [yfinance](https://github.com/ranaroussi/yfinance)
- Black–Scholes Greeks (delta, gamma, theta, vega) with rolling realized volatility
- Band-based delta hedging with configurable sell vol, hedging vol, and recentering
- P&L attribution, realized vol charts, and hedging-vol comparison
- Financing impact tracking (cash and borrow rates) — see [FINANCING_METRICS_GUIDE.md](FINANCING_METRICS_GUIDE.md)

### Monte Carlo Simulation
- Simulate many price paths under configurable realized vol
- Distribution of hedging P&L, error vs. replication benchmark, and sell-vol sweeps
- Financing impact statistics across percentiles

### Efficient Frontier
- Compare delta-value and volatility-price hedging multipliers
- Mean–variance efficient frontier with constraint modes
- Interactive Plotly charts for frontier and unit-multiplier analysis

## Quick Start

### Prerequisites
- Python 3.10 or newer
- pip

### Install and run locally

```bash
git clone https://github.com/kewyang27/Delta-Heding-Backtest.git
cd Delta-Heding-Backtest
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

### Run tests

```bash
python test_hedge_frontier_smoke.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public repo required for the free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and select:
   - **Repository:** `kewyang27/Delta-Heding-Backtest`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **Deploy**.

Streamlit installs dependencies from `requirements.txt` automatically. Pushes to `main` trigger redeploys.

## Project Structure

```
Delta-Heding-Backtest/
├── streamlit_app.py          # Streamlit dashboard (entry point)
├── delta_hedge_backtest.py   # Historical backtest engine
├── mc_delta_hedge_sim.py     # Monte Carlo simulator
├── hedge_frontier/           # Efficient frontier package
│   ├── config.py
│   ├── engine.py
│   ├── hedgers.py
│   ├── optimizer.py
│   ├── metrics.py
│   ├── pricer.py
│   └── viz.py
├── requirements.txt
├── test_hedge_frontier_smoke.py
└── FINANCING_METRICS_GUIDE.md
```

## Dependencies

| Package    | Purpose                          |
| ---------- | -------------------------------- |
| streamlit  | Web dashboard                    |
| numpy      | Numerical computation            |
| pandas     | Data handling                    |
| scipy      | Statistics (normal CDF, etc.)    |
| plotly     | Interactive frontier charts      |
| matplotlib | Backtest and MC visualizations   |
| yfinance   | Historical market data           |
| XlsxWriter | Excel export                     |

## Usage Notes

- **Historical backtest:** Set ticker, strike, dates, and vol parameters in the sidebar, then click **Run Backtest**. Data is downloaded live from Yahoo Finance.
- **Monte Carlo:** Configure path count, horizon, and vol assumptions in Tab 2. Higher `n_paths` values increase runtime.
- **Efficient frontier:** Adjust simulation and sweep settings in Tab 3. Results are cached within the session for faster replays.

## License

This project is provided for educational and research purposes.
