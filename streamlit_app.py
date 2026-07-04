from __future__ import annotations

import math
from typing import Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from io import BytesIO

# Import existing backtest utilities without modifying them
from delta_hedge_backtest import (
    StrategyParams,
    MarketDataFetcher,
    DeltaHedgeBacktester,
)

# Monte Carlo simulator imports
from mc_delta_hedge_sim import (
    MCParams,
    MCDeltaHedgeSimulator,
    sweep_sell_vols,
)

# Efficient frontier hedging imports
from hedge_frontier.config import ConstraintMode, SimulationConfig, SweepConfig, WWRebalanceMode, simulation_config_from_cache_key
from hedge_frontier.metrics import (
    replication_benchmark,
    tracking_std_dev_usd,
    variance_limit_from_std_dev,
)
from hedge_frontier.pricer import OptionPricer
from hedge_frontier.optimizer import FrontierResult, run_full_analysis, compute_baseline_points
from hedge_frontier.viz import (
    build_efficient_frontier_figure,
    build_mean_vs_std_dev_frontier_figure,
    build_unit_multiplier_comparison_figure,
)


st.set_page_config(page_title="Delta Hedging Backtest Dashboard", layout="wide")


def render_header() -> None:
    st.title("Delta Hedging Backtest Dashboard")
    st.caption(
        "Interactive historical backtest with Black–Scholes Greeks, band hedging, and diagnostics."
    )


def inject_center_table_css() -> None:
    """Inject global CSS to center-align Streamlit tables (headers and cells)."""
    st.markdown(
        """
        <style>
        /* Center align all Streamlit static tables */
        [data-testid="stTable"] th, [data-testid="stTable"] td { text-align: center !important; }
        [data-testid="stTable"] table { margin-left: auto; margin-right: auto; }
        /* Fallback for pandas Styler 'dataframe' class */
        .dataframe th, .dataframe td { text-align: center !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _styled_table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> None:
    """Render a center-aligned table using pandas Styler."""
    styler = df.style
    if fmt:
        styler = styler.format(fmt)
    styler = (
        styler.set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "th.row_heading", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ])
    )
    st.table(styler)


def build_params_ui() -> StrategyParams:
    with st.sidebar:
        st.header("Inputs")

        col_a, col_b = st.columns(2)
        with col_a:
            ticker = st.text_input("Ticker", value="SPY")
            strike_price = st.number_input("Strike Price", value=500.0, step=1.0, format="%0.2f")
            risk_free_rate = st.number_input("Risk-free Rate", value=0.05, step=0.005, format="%0.3f")
            option_contracts = st.number_input("Option Contracts (short = negative)", value=-1, step=1)
        with col_b:
            sell_vol = st.number_input("Sell Vol (IV)", value=0.40, step=0.01, format="%0.2f")
            hedging_vol = st.number_input("Hedging Vol (bands)", value=0.30, step=0.01, format="%0.2f")
            greek_vol_lookback = st.number_input("Greek Vol Lookback (days)", value=90, step=5)
            option_type = st.selectbox("Option Type", options=["Call", "Put"], index=0)
            dividend_yield = st.number_input("Dividend Yield (annual)", value=0.00, step=0.005, format="%0.3f")

        col_c, col_d = st.columns(2)
        with col_c:
            hedging_start_date = st.date_input("Hedging Start Date", value=pd.to_datetime("2025-03-01").date())
        with col_d:
            hedging_end_date = st.date_input("Hedging End Date", value=pd.to_datetime("2025-08-31").date())

        st.markdown("---")
        strict_window = st.checkbox("Strict Window (no auto-adjust)", value=True)
        use_greek_vol_for_bands = st.checkbox("Use Realized Vol for Bands (dynamic)", value=False)
        recenter_bands_daily = st.checkbox("Recenter Bands Daily (use yesterday's close)", value=True)
        cash_rate = st.number_input("Cash Rate (annual)", value=0.00, step=0.005, format="%0.3f")
        borrow_rate = st.number_input("Borrow Rate (annual)", value=0.035, step=0.005, format="%0.3f")

        st.markdown("---")
        st.subheader("Comparison Settings")
        compare_hedging_vol = st.number_input("Compare Hedging Vol", value=0.40, step=0.01, format="%0.2f")

        st.markdown("---")
        run_btn = st.button("Run Backtest", type="primary")

    trade_date = pd.to_datetime(hedging_start_date)
    # Align expiry to hedging_end to match hedging window
    expiry_date = pd.to_datetime(hedging_end_date)

    params = StrategyParams(
        ticker=str(ticker),
        strike_price=float(strike_price),
        expiry_date=pd.to_datetime(expiry_date),
        trade_date=pd.to_datetime(trade_date),
        risk_free_rate=float(risk_free_rate),
        option_contracts=int(option_contracts),
        sell_vol=float(sell_vol),
        hedging_vol=float(hedging_vol),
        greek_vol_lookback=int(greek_vol_lookback),
        hedging_start_date=pd.to_datetime(hedging_start_date),
        hedging_end_date=pd.to_datetime(hedging_end_date),
        strict_window=bool(strict_window),
        use_greek_vol_for_bands=bool(use_greek_vol_for_bands),
        recenter_bands_daily=bool(recenter_bands_daily),
        cash_rate=float(cash_rate),
        borrow_rate=float(borrow_rate),
        option_type=str(option_type),
        dividend_yield=float(dividend_yield),
    )

    return params, float(compare_hedging_vol), bool(run_btn)


def plot_cumulative_pnl_and_price(portfolio: pd.DataFrame, data: pd.DataFrame, ticker: str, strike: float) -> None:
    fig, ax_left = plt.subplots(figsize=(12, 5))
    ax_right = ax_left.twinx()
    ax_left.plot(portfolio.index, portfolio["cumulative_pnl"], color="tab:blue", label="Cumulative PnL")
    ax_right.plot(data.index, data["Close"], color="tab:orange", alpha=0.8, label=f"{ticker} Close")
    # Strike level (dotted)
    ax_right.axhline(y=float(strike), color="gray", linestyle=":", linewidth=1.5, label="Strike")
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Cumulative PnL", color="tab:blue")
    ax_right.set_ylabel("Price", color="tab:orange")
    ax_left.grid(True, linestyle="--", alpha=0.3)
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")
    st.pyplot(fig)


def plot_attribution_layers(portfolio: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = portfolio.index
    cum_theta = portfolio["theta_pnl_day"].cumsum()
    cum_gamma = portfolio["gamma_pnl_day"].cumsum()
    cum_vega = portfolio.get("vega_pnl_day", pd.Series(0.0, index=x)).cumsum()
    cum_resid = portfolio["residual_pnl"].cumsum() if "residual_pnl" in portfolio.columns else pd.Series(0.0, index=x)
    cum_total = portfolio["cumulative_pnl"]

    resid_area = ax.fill_between(x, 0, cum_resid, color="#2196F3", alpha=0.30, label="Residual", zorder=1)
    vega_area = ax.fill_between(x, 0, cum_vega, color="#9C27B0", alpha=0.40, label="Vega", zorder=2)
    gamma_area = ax.fill_between(x, 0, cum_gamma, color="#F44336", alpha=0.45, label="Gamma", zorder=3)
    theta_area = ax.fill_between(x, 0, cum_theta, color="#4CAF50", alpha=0.50, label="Theta", zorder=4)
    total_line, = ax.plot(x, cum_total, color="black", linewidth=2.0, label="Total", zorder=5)
    ax.set_title("Cumulative PnL Attribution")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(handles=[theta_area, gamma_area, vega_area, resid_area, total_line], labels=["Theta", "Gamma", "Vega", "Residual", "Total"], loc="best")
    st.pyplot(fig)


def plot_financing_vs_total(portfolio: pd.DataFrame) -> None:
    x = portfolio.index
    cum_fin = portfolio.get("financing_pnl_day", pd.Series(0.0, index=x)).cumsum()
    cum_total = portfolio["cumulative_pnl"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, cum_total, color="black", linewidth=2.0, label="Total")
    ax.plot(x, cum_fin, color="#795548", linewidth=2.0, label="Financing")
    ax.set_title("Cumulative Financing vs Total")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig)


def plot_historical_pnl_financing_impact(portfolio: pd.DataFrame) -> None:
    """Plot and table showing P&L with vs without financing for historical backtest."""
    x = portfolio.index
    cum_fin = portfolio.get("financing_pnl_day", pd.Series(0.0, index=x)).cumsum()
    cum_total = portfolio["cumulative_pnl"]
    cum_ex_fin = cum_total - cum_fin
    
    # Time series plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, cum_total, color="tab:blue", linewidth=2.0, label="P&L with Financing", linestyle="-")
    ax.plot(x, cum_ex_fin, color="tab:green", linewidth=2.0, label="P&L ex-Financing", linestyle="--")
    ax.plot(x, cum_fin, color="tab:red", linewidth=1.5, label="Financing Impact", linestyle=":", alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_title("P&L Comparison: Impact of Financing Over Time (Historical)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig)
    
    # Calculate key metrics
    final_with = float(cum_total.iloc[-1])
    final_ex = float(cum_ex_fin.iloc[-1])
    final_fin_impact = float(cum_fin.iloc[-1])
    
    # Calculate additional metrics
    fin_pnl_day = portfolio.get("financing_pnl_day", pd.Series(0.0, index=x))
    avg_daily_fin = float(fin_pnl_day.mean())
    num_days = len(portfolio)
    
    # Key Financing Ratios
    st.markdown("#### Key Financing Metrics")
    
    fin_to_ex_pnl = (final_fin_impact / final_ex * 100) if final_ex != 0 else 0
    fin_to_with_pnl = (final_fin_impact / final_with * 100) if final_with != 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Financing Cost / Ex-Fin P&L",
            f"{fin_to_ex_pnl:.2f}%",
            help="Total financing impact as % of core strategy P&L (ex-financing)"
        )
    with col2:
        st.metric(
            "Financing Cost / Total P&L",
            f"{fin_to_with_pnl:.2f}%",
            help="Total financing impact as % of final P&L (with financing)"
        )
    with col3:
        st.metric(
            "Total Financing Impact",
            f"${final_fin_impact:,.2f}",
            help="Total financing cost over the backtest period"
        )
    with col4:
        st.metric(
            "Avg Daily Financing",
            f"${avg_daily_fin:,.2f}",
            help=f"Average daily financing cost ({num_days} days)"
        )
    
    # Bar chart for final values
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    categories = ["With Financing", "Ex-Financing", "Financing Impact"]
    values = [final_with, final_ex, final_fin_impact]
    colors = ["tab:blue", "tab:green", "tab:red" if final_fin_impact < 0 else "tab:orange"]
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax2.set_title("Final P&L Comparison", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Final P&L ($)")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}',
                ha='center', va='bottom' if val >= 0 else 'top',
                fontweight='bold', fontsize=10)
    
    st.pyplot(fig2)
    
    # Detailed summary table
    st.markdown("#### Detailed Financing Impact Summary")
    summary_data = {
        "Metric": [
            "Final P&L with Financing",
            "Final P&L ex-Financing",
            "Total Financing Impact",
            "Financing Cost / Ex-Fin P&L",
            "Financing Cost / With-Fin P&L",
            "Average Daily Financing",
            "Max Daily Financing (worst)",
            "Min Daily Financing (best)",
            "Total Backtest Days",
        ],
        "Value": [
            f"${final_with:,.2f}",
            f"${final_ex:,.2f}",
            f"${final_fin_impact:,.2f}",
            f"{fin_to_ex_pnl:.2f}%",
            f"{fin_to_with_pnl:.2f}%",
            f"${avg_daily_fin:,.2f}",
            f"${fin_pnl_day.min():,.2f}",
            f"${fin_pnl_day.max():,.2f}",
            f"{num_days} days",
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    _styled_table(summary_df)


def plot_realized_vol(data: pd.DataFrame, lookback: int, ticker: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.index, data["greek_vol"], color="tab:purple", label=f"{lookback}d realized vol (annualized)")
    ax.set_title(f"{ticker} {lookback}d Realized Volatility (annualized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Realized Vol")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig)


# -------------------------------
# Monte Carlo (Tab 2) components
# -------------------------------

def build_mc_ui() -> tuple[MCParams, list[float], bool, bool]:
    """Build Monte Carlo parameter inputs in Tab 2 (not in sidebar)."""
    st.header("Monte Carlo Delta Hedging Simulator")
    st.caption("Simulate paths and approximate delta-hedging P&L distribution.")

    with st.form("mc_params_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ticker = st.text_input("Ticker", value="SPY", key="mc_ticker")
            S0 = st.number_input("S0 (spot)", value=500.0, step=1.0, format="%0.2f", key="mc_S0")
            strike = st.number_input("Strike", value=500.0, step=1.0, format="%0.2f", key="mc_strike")
            risk_free_rate = st.number_input("Risk-free Rate", value=0.05, step=0.005, format="%0.3f", key="mc_r")
            cash_rate = st.number_input("Cash Rate (annual)", value=0.00, step=0.005, format="%0.3f", key="mc_cash_rate")
        with col2:
            dividend_yield = st.number_input("Dividend Yield", value=0.00, step=0.005, format="%0.3f", key="mc_q")
            option_contracts = st.number_input("Option Contracts (short negative)", value=-1, step=1, key="mc_contracts")
            contract_multiplier = st.number_input("Contract Multiplier", value=100, step=1, key="mc_multiplier")
            rehedge_time_step = st.number_input("Rehedge Time Step (days)", value=1, step=1, key="mc_rehedge")
            borrow_rate = st.number_input("Borrow Rate (annual)", value=0.00, step=0.005, format="%0.3f", key="mc_borrow_rate")
        with col3:
            sell_vol = st.number_input("Sell Vol (IV)", value=0.32, step=0.01, format="%0.2f", key="mc_sellvol")
            path_vol = st.number_input("Path Vol (realized)", value=0.32, step=0.01, format="%0.2f", key="mc_pathvol")
            hedging_vol = st.number_input("Hedging Vol (bands)", value=0.32, step=0.01, format="%0.2f", key="mc_hedgevol")
            use_greek_vol_for_bands = st.checkbox("Use dynamic Greeks vol for bands", value=False, key="mc_use_greek_bands")
            option_type_mc = st.selectbox("Option Type", options=["Call", "Put"], index=0, key="mc_option_type")
        with col4:
            greeks_vol_lookback = st.number_input("Greeks Vol Lookback (days)", value=60, step=5, key="mc_lookback")
            T_years = st.number_input("Tenor in Years", value=0.5, step=0.05, format="%0.2f", key="mc_T")
            steps_per_year = st.number_input("Steps per Year", value=252, step=1, key="mc_steps")
            n_paths = st.number_input("# Monte Carlo Paths", value=2000, step=500, key="mc_npaths")
            seed = st.number_input("Seed (None for random)", value=42, step=1, key="mc_seed")

        st.markdown("---")
        st.subheader("Sweep Settings (sell_vol)")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sweep_min = st.number_input("Min", value=0.20, step=0.01, format="%0.2f", key="mc_sweep_min")
        with col_s2:
            sweep_max = st.number_input("Max", value=0.40, step=0.01, format="%0.2f", key="mc_sweep_max")
        with col_s3:
            sweep_step = st.number_input("Step", value=0.04, step=0.01, format="%0.2f", key="mc_sweep_step")

        run_mc = st.form_submit_button("Run Simulation", type="primary")

    sweep_list = list(np.round(np.arange(sweep_min, sweep_max + 1e-9, sweep_step), 2))

    params = MCParams(
        ticker=str(ticker),
        S0=float(S0),
        strike_price=float(strike),
        risk_free_rate=float(risk_free_rate),
        dividend_yield=float(dividend_yield),
        option_contracts=int(option_contracts),
        sell_vol=float(sell_vol),
        hedging_vol=float(hedging_vol),
        path_vol=float(path_vol),
        greeks_vol_lookback=int(greeks_vol_lookback),
        T_years=float(T_years),
        steps_per_year=int(steps_per_year),
        n_paths=int(n_paths),
        seed=None if seed is None else int(seed),
        contract_multiplier=int(contract_multiplier),
        use_greek_vol_for_bands=bool(use_greek_vol_for_bands),
        rehedge_time_step=int(rehedge_time_step),
        cash_rate=float(cash_rate),
        borrow_rate=float(borrow_rate),
        option_type=str(option_type_mc),
    )

    # Single button triggers both base run and sweep outputs
    run_sweep = run_mc
    return params, sweep_list, bool(run_mc), bool(run_sweep)


def plot_mc_histogram(vals: np.ndarray) -> None:
    mean = float(np.mean(vals))
    p5, p50, p95 = np.percentile(vals, [5, 50, 95])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.hist(vals, bins=50, color="tab:blue", alpha=0.6, edgecolor="white")
    ax.axvline(mean, color="black", linestyle="--", label=f"Mean: {mean:,.0f}")
    ax.axvline(p50, color="tab:green", linestyle=":", label=f"Median: {p50:,.0f}")
    ax.axvspan(p5, p95, color="tab:orange", alpha=0.2, label=f"5th–95th: [{p5:,.0f}, {p95:,.0f}]")
    ax.set_title("Monte Carlo Final Portfolio Value Distribution")
    ax.set_xlabel("Final Portfolio Value")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best")
    st.pyplot(fig)


def plot_mc_histogram_dual(df: pd.DataFrame) -> None:
    with_fin = df["final_portfolio_value"].values
    ex_fin = df.get("final_portfolio_value_ex_fin", pd.Series(np.nan, index=df.index)).values
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.hist(with_fin, bins=50, color="tab:blue", alpha=0.45, edgecolor="white", label="With financing")
    if np.isfinite(ex_fin).all():
        ax.hist(ex_fin, bins=50, color="tab:green", alpha=0.45, edgecolor="white", label="Ex-financing")
    ax.set_title("Monte Carlo Final Portfolio Value: with vs ex-financing")
    ax.set_xlabel("Final Portfolio Value")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best")
    st.pyplot(fig)


def plot_mean_pnl_comparison(df: pd.DataFrame) -> None:
    """Plot and table comparing mean P&L with vs without financing."""
    with_fin = df["final_portfolio_value"].values
    ex_fin = df.get("final_portfolio_value_ex_fin", pd.Series(np.nan, index=df.index)).values
    
    if not np.isfinite(ex_fin).all():
        st.info("Ex-financing data not available.")
        return
    
    # Calculate statistics
    mean_with = float(np.mean(with_fin))
    mean_ex = float(np.mean(ex_fin))
    financing_impact = mean_with - mean_ex
    
    # Bar chart comparing means
    fig, ax = plt.subplots(figsize=(8, 4.5))
    categories = ["With Financing", "Ex-Financing", "Financing Impact"]
    values = [mean_with, mean_ex, financing_impact]
    colors = ["tab:blue", "tab:green", "tab:red" if financing_impact < 0 else "tab:orange"]
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_title("Mean P&L Comparison: Impact of Financing (Monte Carlo)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Final Portfolio Value ($)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}',
                ha='center', va='bottom' if val >= 0 else 'top',
                fontweight='bold', fontsize=10)
    
    st.pyplot(fig)
    
    # Key Financing Ratios
    st.markdown("#### Key Financing Metrics")
    
    # Calculate ratios (handle edge cases)
    fin_to_ex_pnl = (financing_impact / mean_ex * 100) if mean_ex != 0 else 0
    fin_to_with_pnl = (financing_impact / mean_with * 100) if mean_with != 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Financing Cost / Ex-Fin P&L",
            f"{fin_to_ex_pnl:.2f}%",
            help="Financing impact as % of the core strategy P&L (ex-financing)"
        )
    with col2:
        st.metric(
            "Financing Cost / Total P&L",
            f"{fin_to_with_pnl:.2f}%",
            help="Financing impact as % of the final P&L (with financing)"
        )
    with col3:
        st.metric(
            "Mean Financing Impact",
            f"${financing_impact:,.0f}",
            help="Average financing cost across all paths"
        )
    
    # Detailed statistics table
    st.markdown("#### Detailed Financing Impact Analysis")
    stats_data = {
        "Metric": ["Mean", "Median", "Std Dev", "5th Percentile", "95th Percentile"],
        "With Financing": [
            f"${mean_with:,.2f}",
            f"${np.median(with_fin):,.2f}",
            f"${np.std(with_fin):,.2f}",
            f"${np.percentile(with_fin, 5):,.2f}",
            f"${np.percentile(with_fin, 95):,.2f}",
        ],
        "Ex-Financing": [
            f"${mean_ex:,.2f}",
            f"${np.median(ex_fin):,.2f}",
            f"${np.std(ex_fin):,.2f}",
            f"${np.percentile(ex_fin, 5):,.2f}",
            f"${np.percentile(ex_fin, 95):,.2f}",
        ],
        "Impact (Δ)": [
            f"${mean_with - mean_ex:,.2f}",
            f"${np.median(with_fin) - np.median(ex_fin):,.2f}",
            f"${np.std(with_fin) - np.std(ex_fin):,.2f}",
            f"${np.percentile(with_fin, 5) - np.percentile(ex_fin, 5):,.2f}",
            f"${np.percentile(with_fin, 95) - np.percentile(ex_fin, 95):,.2f}",
        ],
    }
    stats_df = pd.DataFrame(stats_data)
    _styled_table(stats_df)
    
    # Additional ratio table
    st.markdown("#### Financing Cost Ratios Across Distribution")
    median_with = float(np.median(with_fin))
    median_ex = float(np.median(ex_fin))
    median_fin = median_with - median_ex
    
    p5_with = float(np.percentile(with_fin, 5))
    p5_ex = float(np.percentile(ex_fin, 5))
    p5_fin = p5_with - p5_ex
    
    p95_with = float(np.percentile(with_fin, 95))
    p95_ex = float(np.percentile(ex_fin, 95))
    p95_fin = p95_with - p95_ex
    
    ratio_data = {
        "Statistic": ["Mean", "Median", "5th Percentile", "95th Percentile"],
        "Fin Cost / Ex-Fin P&L": [
            f"{fin_to_ex_pnl:.2f}%",
            f"{(median_fin / median_ex * 100) if median_ex != 0 else 0:.2f}%",
            f"{(p5_fin / p5_ex * 100) if p5_ex != 0 else 0:.2f}%",
            f"{(p95_fin / p95_ex * 100) if p95_ex != 0 else 0:.2f}%",
        ],
        "Fin Cost / With-Fin P&L": [
            f"{fin_to_with_pnl:.2f}%",
            f"{(median_fin / median_with * 100) if median_with != 0 else 0:.2f}%",
            f"{(p5_fin / p5_with * 100) if p5_with != 0 else 0:.2f}%",
            f"{(p95_fin / p95_with * 100) if p95_with != 0 else 0:.2f}%",
        ],
    }
    ratio_df = pd.DataFrame(ratio_data)
    _styled_table(ratio_df)


def plot_mc_heatmap(sweep_df: pd.DataFrame) -> None:
    display = sweep_df.copy()
    for col in ["mean", "p5", "p50", "p95"]:
        display[col] = display[col].round(0)
    metrics = ["mean", "p5", "p50", "p95"]
    mat = display.set_index("sell_vol")[metrics]
    dv_map = dict(zip(display["sell_vol"], display["dv"]))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    cmap = mpl.cm.get_cmap("RdYlGn")
    vmax = float(np.nanmax(np.abs(mat.values)))
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            ax.text(j, i, f"{val:,.0f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([f"{sv:.2f} (dv {dv_map.get(sv, float('nan')):+.2f})" for sv in mat.index])
    ax.set_xlabel("Metric")
    ax.set_ylabel("sell_vol (with dv)")
    ax.set_title("Sell vol sweep vs fixed path_vol — heatmap ($)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("$", rotation=270, labelpad=12)
    st.pyplot(fig)


def plot_comparison(data: pd.DataFrame, base_params: StrategyParams, base_portfolio: pd.DataFrame, compare_hedging_vol: float, show_ex_fin: bool = True) -> None:
    # Scenario 2: change hedging_vol only (bands from fixed hedging_vol)
    params_cmp = StrategyParams(
        ticker=base_params.ticker,
        strike_price=base_params.strike_price,
        expiry_date=base_params.expiry_date,
        trade_date=base_params.trade_date,
        risk_free_rate=base_params.risk_free_rate,
        option_contracts=base_params.option_contracts,
        sell_vol=base_params.sell_vol,
        hedging_vol=float(compare_hedging_vol),
        greek_vol_lookback=base_params.greek_vol_lookback,
        hedging_start_date=base_params.hedging_start_date,
        hedging_end_date=base_params.hedging_end_date,
        strict_window=base_params.strict_window,
        use_greek_vol_for_bands=False,
        recenter_bands_daily=base_params.recenter_bands_daily,
        cash_rate=base_params.cash_rate,
        borrow_rate=base_params.borrow_rate,
        option_type=base_params.option_type,
        dividend_yield=base_params.dividend_yield,
    )
    backtester_cmp = DeltaHedgeBacktester(params_cmp, data)
    portfolio_cmp = backtester_cmp.run()

    # Scenario 3: dynamic bands using realized vol (rolling realized vol via greek_vol)
    params_dyn = StrategyParams(
        ticker=base_params.ticker,
        strike_price=base_params.strike_price,
        expiry_date=base_params.expiry_date,
        trade_date=base_params.trade_date,
        risk_free_rate=base_params.risk_free_rate,
        option_contracts=base_params.option_contracts,
        sell_vol=base_params.sell_vol,
        hedging_vol=base_params.hedging_vol,
        greek_vol_lookback=base_params.greek_vol_lookback,
        hedging_start_date=base_params.hedging_start_date,
        hedging_end_date=base_params.hedging_end_date,
        strict_window=base_params.strict_window,
        use_greek_vol_for_bands=True,
        recenter_bands_daily=base_params.recenter_bands_daily,
        cash_rate=base_params.cash_rate,
        borrow_rate=base_params.borrow_rate,
        option_type=base_params.option_type,
        dividend_yield=base_params.dividend_yield,
    )
    backtester_dyn = DeltaHedgeBacktester(params_dyn, data)
    portfolio_dyn = backtester_dyn.run()

    fig, ax_left = plt.subplots(figsize=(12, 5))
    ax_right = ax_left.twinx()
    ax_left.plot(base_portfolio.index, base_portfolio["cumulative_pnl"], label=f"Hedge vol {base_params.hedging_vol:.2f}", color="tab:blue")
    ax_left.plot(portfolio_cmp.index, portfolio_cmp["cumulative_pnl"], label=f"Hedge vol {float(compare_hedging_vol):.2f}", color="tab:green")
    ax_left.plot(portfolio_dyn.index, portfolio_dyn["cumulative_pnl"], label=f"Bands = {base_params.greek_vol_lookback}d RV", color="tab:red")
    ax_right.plot(data.index, data["Close"], color="tab:orange", alpha=0.6, label=f"{base_params.ticker} Close")
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Cumulative PnL")
    ax_right.set_ylabel("Price")
    ax_left.grid(True, linestyle="--", alpha=0.3)
    title = f"Cumulative PnL Comparison: Hedge vol {base_params.hedging_vol:.2f} vs {float(compare_hedging_vol):.2f} vs bands={base_params.greek_vol_lookback}d RV (sell vol {base_params.sell_vol:.2f})"
    ax_left.set_title(title)
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Cum PnL (base)", f"{float(base_portfolio['cumulative_pnl'].iloc[-1]):,.2f}")
    with col2:
        st.metric("Final Cum PnL (compare)", f"{float(portfolio_cmp['cumulative_pnl'].iloc[-1]):,.2f}")
    with col3:
        st.metric("Final Cum PnL (bands=RV)", f"{float(portfolio_dyn['cumulative_pnl'].iloc[-1]):,.2f}")

    if show_ex_fin:
        # Overlay ex-fin cumulative PnL (with financing removed)
        def _cum_ex_fin(df: pd.DataFrame) -> pd.Series:
            fin = df.get("financing_pnl_day", pd.Series(0.0, index=df.index)).cumsum()
            return df["cumulative_pnl"] - fin

        base_ex = _cum_ex_fin(base_portfolio)
        cmp_ex = _cum_ex_fin(portfolio_cmp)
        dyn_ex = _cum_ex_fin(portfolio_dyn)

        fig_ex, ax_ex = plt.subplots(figsize=(12, 4))
        ax_ex.plot(base_portfolio.index, base_portfolio["cumulative_pnl"], color="tab:blue", label=f"Base with fin")
        ax_ex.plot(base_portfolio.index, base_ex, color="tab:blue", linestyle="--", label=f"Base ex-fin")
        ax_ex.plot(portfolio_cmp.index, portfolio_cmp["cumulative_pnl"], color="tab:green", label=f"Cmp with fin")
        ax_ex.plot(portfolio_cmp.index, cmp_ex, color="tab:green", linestyle="--", label=f"Cmp ex-fin")
        ax_ex.plot(portfolio_dyn.index, portfolio_dyn["cumulative_pnl"], color="tab:red", label=f"RV with fin")
        ax_ex.plot(portfolio_dyn.index, dyn_ex, color="tab:red", linestyle="--", label=f"RV ex-fin")
        ax_ex.set_title("Cumulative PnL: with vs ex-financing (overlay)")
        ax_ex.set_xlabel("Date")
        ax_ex.set_ylabel("Cumulative PnL")
        ax_ex.grid(True, linestyle="--", alpha=0.3)
        ax_ex.legend(loc="best", ncol=3)
        st.pyplot(fig_ex)

        # Grouped bar: final totals with and without financing
        labels = ["Base", "Compare", "Bands=RV"]
        with_fin = [
            float(base_portfolio["cumulative_pnl"].iloc[-1]),
            float(portfolio_cmp["cumulative_pnl"].iloc[-1]),
            float(portfolio_dyn["cumulative_pnl"].iloc[-1]),
        ]
        ex_fin = [float(base_ex.iloc[-1]), float(cmp_ex.iloc[-1]), float(dyn_ex.iloc[-1])]
        x = np.arange(len(labels))
        width = 0.35
        figb, axb = plt.subplots(figsize=(8, 3.8))
        axb.bar(x - width/2, with_fin, width, label="With fin")
        axb.bar(x + width/2, ex_fin, width, label="Ex-fin")
        axb.set_xticks(x, labels)
        axb.set_title("Final PnL: with vs ex-financing")
        axb.grid(True, axis="y", linestyle="--", alpha=0.3)
        axb.legend()
        st.pyplot(figb)

        # Mini table
        df_stats = pd.DataFrame({
            "Scenario": labels,
            "Final with fin": with_fin,
            "Final ex-fin": ex_fin,
            "Delta fin": [wf - ef for wf, ef in zip(with_fin, ex_fin)],
        })
        df_stats["Final with fin"] = df_stats["Final with fin"].map(lambda v: f"{v:,.2f}")
        df_stats["Final ex-fin"] = df_stats["Final ex-fin"].map(lambda v: f"{v:,.2f}")
        df_stats["Delta fin"] = df_stats["Delta fin"].map(lambda v: f"{v:,.2f}")
        _styled_table(df_stats)

    # Re-hedge counts bar chart
    def _count_rehedges(df: pd.DataFrame) -> int:
        return int((df["stock_holding"].diff().fillna(0) != 0).sum())

    counts = {
        f"Hedge vol {base_params.hedging_vol:.2f}": _count_rehedges(base_portfolio),
        f"Hedge vol {float(compare_hedging_vol):.2f}": _count_rehedges(portfolio_cmp),
        f"Bands {base_params.greek_vol_lookback}d RV": _count_rehedges(portfolio_dyn),
    }
    fig2, ax2 = plt.subplots(figsize=(6, 3.6))
    ax2.bar(list(counts.keys()), list(counts.values()), color=["tab:blue", "tab:green", "tab:red"]) 
    ax2.set_title("Number of Delta Re-hedges (by scenario)")
    ax2.set_ylabel("Count")
    for i, v in enumerate(counts.values()):
        ax2.text(i, v + 0.5, str(v), ha="center", va="bottom")
    st.pyplot(fig2)


@st.cache_data(show_spinner=False)
def cached_frontier_analysis(
    config_key: tuple,
    vol_multipliers: tuple[float, ...],
    gammas: tuple[float, ...],
    mode_value: str,
    limit: float,
) -> dict:
    """Cached analysis without UI progress (used after first successful run)."""
    config = simulation_config_from_cache_key(config_key)
    sweep_config = SweepConfig(
        vol_multipliers=vol_multipliers,
        gammas=gammas,
    )
    mode = ConstraintMode(mode_value)
    result = run_full_analysis(config, sweep_config, mode, limit)
    return {
        "vol_sweep": result.vol_sweep,
        "gamma_sweep": result.gamma_sweep,
        "optimal_vol": result.optimal_vol,
        "optimal_gamma": result.optimal_gamma,
        "baseline_vol": result.baseline_vol,
        "baseline_gamma": result.baseline_gamma,
    }


def _frontier_payload_hash(payload: dict) -> str:
    return str(
        (
            payload["config_key"],
            payload["vol_multipliers"],
            payload["gammas"],
            payload["mode_value"],
            payload["limit"],
        )
    )


def _build_frontier_config_from_session() -> tuple[SimulationConfig, SweepConfig, ConstraintMode, float, float | None] | None:
    payload = st.session_state.get("ef_run_payload")
    if not payload:
        return None
    config = simulation_config_from_cache_key(payload["config_key"])
    sweep_config = SweepConfig(
        vol_multipliers=tuple(payload["vol_multipliers"]),
        gammas=tuple(payload["gammas"]),
    )
    mode = ConstraintMode(payload["mode_value"])
    limit = float(payload["limit"])
    limit_rmse = payload.get("limit_rmse_usd")
    limit_rmse_usd = None if limit_rmse is None else float(limit_rmse)
    return config, sweep_config, mode, limit, limit_rmse_usd


def build_frontier_ui() -> bool:
    st.subheader("Market Inputs")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        S0 = st.number_input("Initial Spot (S₀)", value=100.0, step=1.0, format="%0.2f", key="ef_S0")
        sigma = st.number_input("Implied Volatility (σ)", value=0.30, step=0.01, format="%0.2f", key="ef_sigma")
        T_days = st.number_input("Days to Maturity", value=63, step=1, key="ef_Tdays")
    with col_b:
        strike = st.number_input("Strike (K)", value=100.0, step=1.0, format="%0.2f", key="ef_strike")
        risk_free_rate = st.number_input("Risk-free Rate (r)", value=0.05, step=0.005, format="%0.3f", key="ef_r")
        k = st.number_input("Transaction Cost (k)", value=0.001, step=0.0005, format="%0.4f", key="ef_k")
    with col_c:
        n_paths = st.number_input("# Simulation Paths", value=1000, step=250, key="ef_npaths")
        seed_input = st.number_input("Seed (0 = random)", value=42, step=1, key="ef_seed")
        steps_per_year = st.number_input("Steps per Year", value=252, step=1, key="ef_steps_py")
        contract_multiplier = st.number_input("Contract Multiplier", value=100, step=1, key="ef_mult")

    st.markdown("---")
    st.subheader("Whalley–Wilmott Rebalance")
    ww_mode_label = st.radio(
        "On delta-band breach, rebalance to",
        options=[
            "Band edge (WW default — lower cost, residual delta inside band)",
            "Full BSM delta (match vol-band breach behavior)",
        ],
        key="ef_ww_rebalance",
        horizontal=True,
    )
    ww_rebalance_mode = (
        WWRebalanceMode.HEDGE_TO_EDGE
        if ww_mode_label.startswith("Band edge")
        else WWRebalanceMode.HEDGE_TO_BSM_DELTA
    )

    st.markdown("---")
    st.subheader("Desk Constraints")
    col_d, col_e = st.columns(2)
    with col_d:
        constraint_label = st.radio(
            "Optimize by",
            options=["Maximum Allowable Tracking Error Std Dev ($)", "Maximum Allowable Transaction Cost"],
            key="ef_constraint_mode",
        )
    with col_e:
        mode = (
            ConstraintMode.MAX_VARIANCE
            if "Tracking Error Std Dev" in constraint_label
            else ConstraintMode.MAX_COST
        )
        if mode == ConstraintMode.MAX_VARIANCE:
            limit_std_dev_usd = st.number_input(
                "Max Tracking Error Std Dev ($)",
                value=150.0,
                step=50.0,
                min_value=0.0,
                format="%0.2f",
                key="ef_var_rmse_limit",
                help="Typical path-to-path spread of tracking errors (√variance). Compare to BSM premium below.",
            )
            limit = variance_limit_from_std_dev(limit_std_dev_usd)
            T_years = int(T_days) / int(steps_per_year)
            bsm_premium, bsm_per_share = OptionPricer.position_premium_usd(
                S0=float(S0),
                strike=float(strike),
                T_years=T_years,
                risk_free_rate=float(risk_free_rate),
                sigma=float(sigma),
                option_contracts=-1,
                contract_multiplier=int(contract_multiplier),
            )
            pct_of_premium = (limit_std_dev_usd / bsm_premium * 100.0) if bsm_premium > 0 else 0.0
            st.caption(
                f"**BSM premium sold (benchmark):** **${bsm_premium:,.2f}** "
                f"(${bsm_per_share:.2f}/share × {int(contract_multiplier):,} shares). "
                f"Std dev limit = **{pct_of_premium:.1f}%** of that premium. "
                "Positive mean tracking = hedge replication beats BSM; negative = shortfall vs BSM."
            )
        else:
            limit_std_dev_usd = None
            limit = st.number_input(
                "Max Transaction Cost ($)",
                value=500.0,
                step=50.0,
                format="%0.2f",
                key="ef_cost_limit",
            )

    with st.expander("Advanced Sweep Settings"):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            vol_min = st.number_input("Vol Multiplier Min", value=0.1, step=0.1, format="%0.2f", key="ef_vol_min")
            vol_max = st.number_input("Vol Multiplier Max", value=3.0, step=0.1, format="%0.2f", key="ef_vol_max")
            vol_steps = st.number_input("Vol Multiplier Steps", value=15, step=1, key="ef_vol_steps")
        with col_s2:
            gamma_min_exp = st.number_input("Gamma Min (10^x)", value=-2.0, step=0.5, format="%0.1f", key="ef_gamma_min")
            gamma_max_exp = st.number_input("Gamma Max (10^x)", value=2.0, step=0.5, format="%0.1f", key="ef_gamma_max")
            gamma_steps = st.number_input("Gamma Steps", value=20, step=1, key="ef_gamma_steps")

    run_frontier = st.button("Run Efficient Frontier", type="primary", key="ef_run_btn")

    vol_multipliers = tuple(float(x) for x in np.linspace(vol_min, vol_max, int(vol_steps)))
    gammas = tuple(float(x) for x in np.logspace(gamma_min_exp, gamma_max_exp, int(gamma_steps)))
    config = SimulationConfig(
        S0=float(S0),
        strike=float(strike),
        sigma=float(sigma),
        risk_free_rate=float(risk_free_rate),
        T_days=int(T_days),
        transaction_cost=float(k),
        n_paths=int(n_paths),
        seed=None if int(seed_input) == 0 else int(seed_input),
        steps_per_year=int(steps_per_year),
        contract_multiplier=int(contract_multiplier),
        ww_rebalance_mode=ww_rebalance_mode,
    )

    if run_frontier:
        st.session_state["ef_run_payload"] = {
            "config_key": config.cache_key(),
            "vol_multipliers": vol_multipliers,
            "gammas": gammas,
            "mode_value": mode.value,
            "limit": float(limit),
            "limit_rmse_usd": None if limit_std_dev_usd is None else float(limit_std_dev_usd),
        }
        st.session_state.pop("ef_result", None)
        st.session_state.pop("ef_result_hash", None)

    return bool(run_frontier)


def render_frontier_tab() -> None:
    st.header("Efficient Frontier: Delta Hedging Strategies")
    st.caption(
        "Compare daily σ√Δt price bands (recentered on prior close) vs Whalley-Wilmott delta-band hedging "
        "on replication consistency (tracking error std dev in $) vs transaction costs."
    )

    submitted = build_frontier_ui()
    run_payload = _build_frontier_config_from_session()

    if run_payload is None:
        st.info("Configure inputs above and click 'Run Efficient Frontier'.")
        return

    config, sweep_config, mode, limit, limit_rmse_usd = run_payload
    payload_hash = _frontier_payload_hash(st.session_state["ef_run_payload"])
    total_sweeps = len(sweep_config.vol_multipliers) + len(sweep_config.gammas)

    if submitted:
        st.caption(
            f"Running {total_sweeps} parameter sweeps across {config.n_paths:,} paths "
            f"({config.T_days} trading days). First run typically takes 30–90 seconds."
        )

    if st.session_state.get("ef_result_hash") == payload_hash and "ef_result" in st.session_state:
        cached = st.session_state["ef_result"]
    else:
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def _report_progress(step: int, total: int, label: str) -> None:
            progress_bar.progress(step / total)
            status_box.info(f"{label} — step {step} of {total}")

        try:
            result = run_full_analysis(
                config,
                sweep_config,
                mode,
                limit,
                progress=_report_progress,
            )
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            return

        cached = {
            "vol_sweep": result.vol_sweep,
            "gamma_sweep": result.gamma_sweep,
            "optimal_vol": result.optimal_vol,
            "optimal_gamma": result.optimal_gamma,
            "baseline_vol": result.baseline_vol,
            "baseline_gamma": result.baseline_gamma,
        }
        st.session_state["ef_result"] = cached
        st.session_state["ef_result_hash"] = payload_hash
        progress_bar.progress(1.0)
        status_box.success("Simulation complete.")

    vol_df = cached["vol_sweep"]
    gamma_df = cached["gamma_sweep"]
    optimal_vol = cached["optimal_vol"]
    optimal_gamma = cached["optimal_gamma"]
    baseline_vol = cached.get("baseline_vol")
    baseline_gamma = cached.get("baseline_gamma")

    st.subheader("Optimization Output")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if optimal_vol.feasible:
            st.metric("Optimal Vol Multiplier", f"{optimal_vol.parameter:.3f}")
            st.caption(
                f"Cost: ${optimal_vol.mean_cost:,.2f} | "
                f"Std Dev: ${tracking_std_dev_usd(optimal_vol.var_error):,.0f}"
            )
        else:
            st.metric("Optimal Vol Multiplier", "N/A")
            st.caption("No feasible point under constraint")
    with col2:
        if optimal_gamma.feasible:
            st.metric("Optimal Risk Aversion (γ)", f"{optimal_gamma.parameter:.4f}")
            st.caption(
                f"Cost: ${optimal_gamma.mean_cost:,.2f} | "
                f"Std Dev: ${tracking_std_dev_usd(optimal_gamma.var_error):,.0f}"
            )
        else:
            st.metric("Optimal Risk Aversion (γ)", "N/A")
            st.caption("No feasible point under constraint")
    with col3:
        constraint_name = (
            "Max Tracking Error Std Dev ($)"
            if mode == ConstraintMode.MAX_VARIANCE
            else "Max Transaction Cost"
        )
        limit_label = (
            f"${limit_rmse_usd:,.0f} std dev"
            if mode == ConstraintMode.MAX_VARIANCE and limit_rmse_usd is not None
            else f"${limit:,.2f}"
        )
        st.metric("Constraint Mode", constraint_name)
        st.caption(f"Limit: {limit_label}")
    with col4:
        if optimal_vol.feasible and optimal_gamma.feasible:
            st.success(
                "To meet constraints, set Volatility Multiplier = "
                f"{optimal_vol.parameter:.3f} OR set Delta Band γ = {optimal_gamma.parameter:.4f}."
            )
        elif optimal_vol.feasible:
            st.warning(
                f"Vol strategy feasible (multiplier={optimal_vol.parameter:.3f}); "
                "delta-band strategy has no feasible point."
            )
        elif optimal_gamma.feasible:
            st.warning(
                f"Delta-band strategy feasible (γ={optimal_gamma.parameter:.4f}); "
                "vol-price strategy has no feasible point."
            )
        else:
            st.error("Neither strategy has a feasible point under the selected constraint.")

    if baseline_vol is None or baseline_gamma is None:
        baseline_vol, baseline_gamma = compute_baseline_points(config, sweep_config)

    result = FrontierResult(
        vol_sweep=vol_df,
        gamma_sweep=gamma_df,
        optimal_vol=optimal_vol,
        optimal_gamma=optimal_gamma,
        baseline_vol=baseline_vol,
        baseline_gamma=baseline_gamma,
    )
    bsm_premium_usd = OptionPricer.position_premium_usd(
        S0=config.S0,
        strike=config.strike,
        T_years=config.T_years,
        risk_free_rate=config.risk_free_rate,
        sigma=config.sigma,
        option_contracts=config.option_contracts,
        contract_multiplier=config.contract_multiplier,
        dividend_yield=config.dividend_yield,
    )[0]
    benchmark = replication_benchmark(bsm_premium_usd, baseline_vol, baseline_gamma)

    st.subheader("Replication vs BSM Benchmark")
    st.caption(
        "You sold the option at the BSM premium. At expiry, effective replication value "
        "= BSM premium + tracking error. Example: sold at $658, replication $600 → "
        "tracking error −$58 (against you); replication $700 → +$42 (favorable)."
    )
    bcol0, bcol1, bcol2 = st.columns(3)
    with bcol0:
        st.metric("BSM Premium Sold (benchmark)", f"${benchmark.bsm_premium_usd:,.2f}")
    with bcol1:
        vol_rep = benchmark.vol_band.replication_value_usd(benchmark.bsm_premium_usd)
        st.metric(
            "Vol Band (mult = 1) — replication",
            f"${vol_rep:,.2f}",
            delta=f"Tracking {benchmark.vol_band.tracking_label}",
        )
        st.caption(f"Tracking std dev: ${benchmark.vol_band.tracking_std_dev:,.2f}")
    with bcol2:
        delta_rep = benchmark.delta_band.replication_value_usd(benchmark.bsm_premium_usd)
        st.metric(
            "Delta Band (γ = 1) — replication",
            f"${delta_rep:,.2f}",
            delta=f"Tracking {benchmark.delta_band.tracking_label}",
        )
        st.caption(f"Tracking std dev: ${benchmark.delta_band.tracking_std_dev:,.2f}")

    fig = build_efficient_frontier_figure(
        result,
        mode,
        limit,
        limit_std_dev_usd=limit_rmse_usd,
        ww_rebalance_mode=config.ww_rebalance_mode,
        bsm_premium_usd=bsm_premium_usd,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mean Tracking vs Replication Consistency")
    st.caption(
        "X-axis = replication consistency (lower std dev is better). "
        "Y-axis = average tracking vs BSM (higher is better). "
        "Ideal is top-left: low std dev with favorable mean tracking."
    )
    mean_std_fig = build_mean_vs_std_dev_frontier_figure(
        result,
        ww_rebalance_mode=config.ww_rebalance_mode,
        limit_std_dev_usd=limit_rmse_usd if mode == ConstraintMode.MAX_VARIANCE else None,
    )
    st.plotly_chart(mean_std_fig, use_container_width=True)

    st.subheader("Baseline Comparison (Multiplier = 1)")
    st.caption(
        "Head-to-head at vol band multiplier = 1.0 and Whalley–Wilmott risk aversion γ = 1.0 "
        f"(WW rebalance: {'edge' if config.ww_rebalance_mode == WWRebalanceMode.HEDGE_TO_EDGE else 'full BSM delta'})."
    )
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.metric(
            "Vol Price Band (mult = 1) — txn cost",
            f"${baseline_vol.mean_cost:,.2f}",
            delta=f"Std dev ${benchmark.vol_band.tracking_std_dev:,.0f}",
        )
    with bcol2:
        st.metric(
            "Delta Band (γ = 1) — txn cost",
            f"${baseline_gamma.mean_cost:,.2f}",
            delta=f"Std dev ${benchmark.delta_band.tracking_std_dev:,.0f}",
        )
    baseline_fig = build_unit_multiplier_comparison_figure(
        baseline_vol,
        baseline_gamma,
        ww_rebalance_mode=config.ww_rebalance_mode,
        bsm_premium_usd=bsm_premium_usd,
    )
    st.plotly_chart(baseline_fig, use_container_width=True)

    with st.expander("Sweep Detail Tables"):
        vol_display = vol_df.assign(
            tracking_std_dev=vol_df["var_error"].apply(tracking_std_dev_usd)
        )[
            ["spot_multiplier", "mean_cost", "tracking_std_dev", "mean_error"]
        ]
        gamma_display = gamma_df.assign(
            tracking_std_dev=gamma_df["var_error"].apply(tracking_std_dev_usd)
        )[
            ["gamma", "mean_cost", "tracking_std_dev", "mean_error"]
        ]

        st.markdown("**Volatility Price Band Sweep**")
        _styled_table(
            vol_display,
            fmt={
                "spot_multiplier": "{:.3f}",
                "mean_cost": "${:,.2f}",
                "tracking_std_dev": "${:,.2f}",
                "mean_error": "${:,.2f}",
            },
        )
        st.markdown("**Delta Value Band Sweep**")
        _styled_table(
            gamma_display,
            fmt={
                "gamma": "{:.4f}",
                "mean_cost": "${:,.2f}",
                "tracking_std_dev": "${:,.2f}",
                "mean_error": "${:,.2f}",
            },
        )


def main() -> None:
    render_header()
    inject_center_table_css()
    tab1, tab2, tab3 = st.tabs(
        ["Historical Backtest", "Monte Carlo Simulation", "Efficient Frontier"]
    )

    # ------------------ Tab 1: Keep existing behavior ------------------
    with tab1:
        params, compare_hedging_vol, run_btn = build_params_ui()

        if not run_btn:
            st.info("Set inputs in the sidebar and click 'Run Backtest'.")
        else:
            try:
                fetcher = MarketDataFetcher(params)
                data = fetcher.fetch_and_prepare()
            except Exception as exc:
                st.error(f"Data preparation failed: {exc}")
            else:
                backtester = DeltaHedgeBacktester(params, data)
                portfolio = backtester.run()

                st.subheader("1) Cumulative PnL vs Price")
                plot_cumulative_pnl_and_price(portfolio, data, params.ticker, params.strike_price)

                st.subheader("2) PnL Attribution (Theta/Gamma/Vega/Residual)")
                plot_attribution_layers(portfolio)

                st.subheader("3) Realized Volatility")
                plot_realized_vol(data, params.greek_vol_lookback, params.ticker)

                st.subheader("4) Hedging Vol Comparison")
                plot_comparison(data, params, portfolio, compare_hedging_vol, show_ex_fin=True)

                # Guidance if financing is zero and ex-fin lines overlap
                if float(params.cash_rate) == 0.0 and float(params.borrow_rate) == 0.0:
                    st.info("Financing rates are zero; 'with' and 'ex-fin' series will overlap. Set non-zero cash/borrow rates to see differences.")

                st.subheader("5) Financing vs Total")
                plot_financing_vs_total(portfolio)

                st.subheader("6) P&L Impact of Financing")
                plot_historical_pnl_financing_impact(portfolio)

                st.markdown("---")
                with st.expander("Show last 5 rows of portfolio"):
                    _styled_table(portfolio.tail().reset_index())

                # Download buttons for Excel export (base and comparison runs)
                st.markdown("---")
                st.subheader("Export Results")
                try:
                    # Reuse comparison portfolios for export
                    cmp_params = StrategyParams(
                        ticker=params.ticker,
                        strike_price=params.strike_price,
                        expiry_date=params.expiry_date,
                        trade_date=params.trade_date,
                        risk_free_rate=params.risk_free_rate,
                        option_contracts=params.option_contracts,
                        sell_vol=params.sell_vol,
                        hedging_vol=float(compare_hedging_vol),
                        greek_vol_lookback=params.greek_vol_lookback,
                        hedging_start_date=params.hedging_start_date,
                        hedging_end_date=params.hedging_end_date,
                        strict_window=params.strict_window,
                        use_greek_vol_for_bands=False,
                        recenter_bands_daily=params.recenter_bands_daily,
                        cash_rate=params.cash_rate,
                        borrow_rate=params.borrow_rate,
                        option_type=params.option_type,
                        dividend_yield=params.dividend_yield,
                    )
                    backtester_cmp = DeltaHedgeBacktester(cmp_params, data)
                    portfolio_cmp = backtester_cmp.run()

                    dyn_params = StrategyParams(
                        ticker=params.ticker,
                        strike_price=params.strike_price,
                        expiry_date=params.expiry_date,
                        trade_date=params.trade_date,
                        risk_free_rate=params.risk_free_rate,
                        option_contracts=params.option_contracts,
                        sell_vol=params.sell_vol,
                        hedging_vol=params.hedging_vol,
                        greek_vol_lookback=params.greek_vol_lookback,
                        hedging_start_date=params.hedging_start_date,
                        hedging_end_date=params.hedging_end_date,
                        strict_window=params.strict_window,
                        use_greek_vol_for_bands=True,
                        recenter_bands_daily=params.recenter_bands_daily,
                        cash_rate=params.cash_rate,
                        borrow_rate=params.borrow_rate,
                        option_type=params.option_type,
                        dividend_yield=params.dividend_yield,
                    )
                    backtester_dyn = DeltaHedgeBacktester(dyn_params, data)
                    portfolio_dyn = backtester_dyn.run()

                    start_str = pd.to_datetime(params.trade_date).strftime('%Y%m%d')
                    end_str = pd.to_datetime(params.expiry_date).strftime('%Y%m%d')
                    filename = f"backtest_{params.ticker}_{start_str}_{end_str}.xlsx"

                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        portfolio.to_excel(writer, sheet_name="portfolio_base")
                        portfolio_cmp.to_excel(writer, sheet_name="portfolio_cmp")
                        portfolio_dyn.to_excel(writer, sheet_name="portfolio_dyn")
                        data.to_excel(writer, sheet_name="market_data")
                    st.download_button(
                        label=f"Download Excel ({filename})",
                        data=output.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as exc:
                    st.warning(f"Excel export unavailable: {exc}")

    # ------------------ Tab 2: Monte Carlo Simulation ------------------
    with tab2:
        mc_params, sweep_list, run_mc, run_sweep = build_mc_ui()

        if run_mc:
            sim = MCDeltaHedgeSimulator(mc_params)
            df = sim.run()
            vals = df["final_portfolio_value"].values
            st.subheader("Monte Carlo — Base Scenario Summary")
            base_stats = pd.Series(vals).describe(percentiles=[0.05, 0.5, 0.95]).rename("final_portfolio_value")
            _styled_table(base_stats.to_frame())
            st.subheader("Distribution")
            plot_mc_histogram(vals)
            if "final_portfolio_value_ex_fin" in df.columns:
                st.subheader("Distribution: with vs ex-financing")
                plot_mc_histogram_dual(df)
                st.markdown("---")
                st.subheader("Mean P&L: Financing Impact")
                plot_mean_pnl_comparison(df)
            # Persist last successful params for export on rerun
            st.session_state["mc_last_params"] = mc_params

        if run_sweep:
            st.subheader("Sell Vol Sweep Results")
            sweep_df = sweep_sell_vols(mc_params, sweep_list)
            _styled_table(
                sweep_df,
                fmt={
                    "sell_vol": "{:.2f}",
                    "dv": "+{:.2f}",
                    "mean": "{:,.0f}",
                    "p5": "{:,.0f}",
                    "p50": "{:,.0f}",
                    "p95": "{:,.0f}",
                },
            )
            st.subheader("Heatmap")
            plot_mc_heatmap(sweep_df)

        # Export block placed outside run_mc so it survives reruns triggered by button clicks
        st.markdown("---")
        st.subheader("Export Simulation Details")
        if "mc_last_params" not in st.session_state:
            st.info("Run Simulation above, then generate the export.")
        else:
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                quantiles = st.multiselect(
                    "Sample quantiles for path timeseries",
                    options=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
                    default=[0.05, 0.50, 0.95],
                    key="mc_export_quantiles",
                )
                include_events = st.checkbox("Include events Parquet (all paths)", value=False, key="mc_export_events")
            with exp_col2:
                do_export = st.button("Generate Export Files", type="secondary", key="mc_do_export")

            if do_export:
                last_params = st.session_state["mc_last_params"]
                sim2 = MCDeltaHedgeSimulator(last_params)
                details = sim2.run_with_details(sample_quantiles=quantiles, return_events=include_events)
                # Build Excel
                excel_name = f"mc_export_{last_params.ticker}_T{last_params.T_years:.2f}_paths{last_params.n_paths}.xlsx"
                output = BytesIO()
                try:
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        pd.DataFrame({
                            "param": list(vars(last_params).keys()),
                            "value": list(vars(last_params).values()),
                        }).to_excel(writer, sheet_name="parameters", index=False)
                        details["summary"].to_excel(writer, sheet_name="summary", index=False)
                        details["finals"].to_excel(writer, sheet_name="path_finals", index=False)
                        if details.get("samples") is not None:
                            details["samples"].to_excel(writer, sheet_name="samples", index=False)
                    st.download_button(
                        label=f"Download Excel ({excel_name})",
                        data=output.getvalue(),
                        file_name=excel_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="mc_download_excel",
                    )
                except Exception as exc:
                    st.warning(f"Excel export unavailable: {exc}. Try installing 'XlsxWriter': pip install XlsxWriter")

    # ------------------ Tab 3: Efficient Frontier ------------------
    with tab3:
        render_frontier_tab()


if __name__ == "__main__":
    main()


