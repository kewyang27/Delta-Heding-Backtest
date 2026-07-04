from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
from io import BytesIO


# -----------------------------
# Step 1: Strategy Parameters
# -----------------------------


@dataclass
class StrategyParams:
    ticker: str
    strike_price: float
    expiry_date: pd.Timestamp
    trade_date: pd.Timestamp
    risk_free_rate: float
    option_contracts: int  # negative for short
    sell_vol: float  # implied vol used to sell the option
    hedging_vol: float  # vol used to set hedge band width
    greek_vol_lookback: int  # days
    contract_multiplier: int = 100  # US equity options
    trading_days_per_year: int = 252
    option_type: str = "Call"  # "Call" or "Put"
    dividend_yield: float = 0.0  # continuous dividend yield q
    # New: explicit hedging window and tenor controls
    hedging_start_date: pd.Timestamp | None = None
    hedging_end_date: pd.Timestamp | None = None
    strict_window: bool = True  # if True, do not auto-adjust to available dates
    # New: use realized greek_vol for hedge band widths instead of fixed hedging_vol
    use_greek_vol_for_bands: bool = False
    # New: recenter hedge bands daily off yesterday's close using hedging_vol
    recenter_bands_daily: bool = True
    # Financing rates
    cash_rate: float = 0.00  # earned on positive cash
    borrow_rate: float = 0.035  # paid on negative cash


# -----------------------------------
# Step 3: Black-Scholes Helper Class
# -----------------------------------


class BlackScholes:
    @staticmethod
    def _ensure_positive_sigma(sigma: float) -> float:
        return max(1e-8, float(sigma))

    @staticmethod
    def _ensure_positive_T(T: float) -> float:
        return max(0.0, float(T))

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        T = BlackScholes._ensure_positive_T(T)
        if T == 0.0:
            return max(S - K, 0.0)
        sigma = BlackScholes._ensure_positive_sigma(sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def call_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        T = BlackScholes._ensure_positive_T(T)
        if T == 0.0:
            # At expiration, delta is 1 if ITM, else 0 for a call
            return 1.0 if S > K else 0.0
        sigma = BlackScholes._ensure_positive_sigma(sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        return float(math.exp(-q * T) * norm.cdf(d1))

    @staticmethod
    def call_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        T = BlackScholes._ensure_positive_T(T)
        if T == 0.0:
            return 0.0
        sigma = BlackScholes._ensure_positive_sigma(sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        return float(math.exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrtT))

    @staticmethod
    def call_theta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        # Returns theta per calendar year (not per day). Convert as needed by /252
        T = BlackScholes._ensure_positive_T(T)
        if T == 0.0:
            return 0.0
        sigma = BlackScholes._ensure_positive_sigma(sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        term1 = - (S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2.0 * sqrtT)
        term2 = - r * K * math.exp(-r * T) * norm.cdf(d2)
        term3 = + q * S * math.exp(-q * T) * norm.cdf(d1)
        return float(term1 + term2 + term3)

    @staticmethod
    def call_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        # Vega is dPrice/dSigma (per 1.0 change in sigma)
        T = BlackScholes._ensure_positive_T(T)
        if T == 0.0:
            return 0.0
        sigma = BlackScholes._ensure_positive_sigma(sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        return float(S * math.exp(-q * T) * norm.pdf(d1) * sqrtT)

    # --------- Put formulas ---------
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        T = BlackScholes._ensure_positive_T(T)
        if T == 0.0:
            return max(K - S, 0.0)
        sigma = BlackScholes._ensure_positive_sigma(sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        T = BlackScholes._ensure_positive_T(T)
        if T == 0.0:
            return -1.0 if S < K else 0.0
        sigma = BlackScholes._ensure_positive_sigma(sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        return float(math.exp(-q * T) * (norm.cdf(d1) - 1.0))

    @staticmethod
    def put_theta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        # Annual theta for puts
        T = BlackScholes._ensure_positive_T(T)
        if T == 0.0:
            return 0.0
        sigma = BlackScholes._ensure_positive_sigma(sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        term1 = - (S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2.0 * sqrtT)
        term2 = + r * K * math.exp(-r * T) * norm.cdf(-d2)
        term3 = - q * S * math.exp(-q * T) * norm.cdf(-d1)
        return float(term1 + term2 + term3)


# ---------------------------------------------
# Step 2: Data Fetching and Preparation Utility
# ---------------------------------------------


class MarketDataFetcher:
    def __init__(self, params: StrategyParams):
        self.params = params

    def fetch_and_prepare(self) -> pd.DataFrame:
        # Add an extended warmup window so rolling vol is populated on TRADE_DATE
        warmup_days = int(self.params.greek_vol_lookback * 3)
        # Use hedging window if provided; otherwise default to trade->expiry
        hedge_start = pd.to_datetime(self.params.hedging_start_date or self.params.trade_date)
        hedge_end = pd.to_datetime(self.params.hedging_end_date or self.params.expiry_date)

        start_date = (hedge_start - pd.Timedelta(days=warmup_days)).date()
        end_date = hedge_end.date()

        data = yf.download(
            self.params.ticker,
            start=start_date,
            end=end_date + pd.Timedelta(days=1),  # inclusive end
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            raise RuntimeError("No data fetched from Yahoo Finance. Check ticker or date range.")

        # Keep only the Close for simplicity
        df = data[["Close"]].copy()
        df.dropna(inplace=True)

        # Daily log returns
        df["log_return"] = np.log(df["Close"]).diff()

        # Rolling historical volatility (annualized)
        lookback = int(self.params.greek_vol_lookback)
        df["greek_vol"] = (
            df["log_return"].rolling(window=lookback, min_periods=lookback).std()
            * math.sqrt(self.params.trading_days_per_year)
        )

        # Handle NaNs: back/forward fill then fall back to hedging_vol
        df["greek_vol"] = df["greek_vol"].bfill().ffill().fillna(float(self.params.hedging_vol))

        # Determine effective window based on hedging window
        if self.params.strict_window:
            effective_start = hedge_start
            effective_end = hedge_end
        else:
            first_available = df.index.min()
            last_available = df.index.max()
            idx_ge_trade = df.index[df.index >= hedge_start]
            effective_start = idx_ge_trade.min() if len(idx_ge_trade) > 0 else last_available
            effective_end = min(hedge_end, last_available)
            if effective_start != hedge_start:
                print(
                    f"Warning: Adjusting hedging start from {hedge_start.date()} to {effective_start.date()}."
                )
            if effective_end != hedge_end:
                print(
                    f"Warning: Adjusting hedging end from {hedge_end.date()} to {effective_end.date()}."
                )

        df = df.loc[(df.index >= effective_start) & (df.index <= effective_end)].copy()

        if df.empty:
            raise RuntimeError(
                "No data in the hedging window after trimming. "
                "Set strict_window=False to auto-adjust or change dates."
            )

        return df


# ------------------------------------------------
# Step 4-6: OOP Backtester with Hedging and PnL
# ------------------------------------------------


class DeltaHedgeBacktester:
    def __init__(self, params: StrategyParams, market_data: pd.DataFrame):
        self.params = params
        self.data = market_data
        self.portfolio = pd.DataFrame(index=self.data.index)
        for col in [
            "cash",
            "stock_holding",
            "stock_value",
            "option_value",
            "portfolio_value",
            "pnl",
            "option_delta",
            "option_gamma",
            "option_theta_per_day",
            "option_vega",
            "theta_pnl_day",
            "gamma_pnl_day",
            "vega_pnl_day",
            "financing_pnl_day",
            "residual_pnl",
        ]:
            self.portfolio[col] = 0.0

        # Hedge bands
        self.upper_band: float | None = None
        self.lower_band: float | None = None

    def _year_fraction_to_expiry(self, current_date: pd.Timestamp) -> float:
        days = (pd.to_datetime(self.params.expiry_date) - pd.to_datetime(current_date)).days
        return max(0.0, days / 365.0)

    def _reset_bands(self, price: float, vol_for_bands: float | None = None) -> None:
        band_vol = float(self.params.hedging_vol) if vol_for_bands is None else float(vol_for_bands)
        daily_move_pct = band_vol / math.sqrt(self.params.trading_days_per_year)
        self.upper_band = price * (1.0 + daily_move_pct)
        self.lower_band = price * (1.0 - daily_move_pct)

    def run(self) -> pd.DataFrame:
        K = float(self.params.strike_price)
        r = float(self.params.risk_free_rate)
        contracts = int(self.params.option_contracts)
        multiplier = int(self.params.contract_multiplier)

        # Initialize on the first day
        first_date = self.data.index[0]
        S0 = float(self.data.loc[first_date, "Close"].iloc[0] if isinstance(self.data.loc[first_date, "Close"], pd.Series) else self.data.loc[first_date, "Close"])
        vol0 = float(self.data.loc[first_date, "greek_vol"].iloc[0] if isinstance(self.data.loc[first_date, "greek_vol"], pd.Series) else self.data.loc[first_date, "greek_vol"])
        T0 = self._year_fraction_to_expiry(first_date)

        cash = 0.0
        stock_holding = 0.0

        # Sell/buy the option at sell_vol, receive/pay premium
        if self.params.option_type.lower() == "put":
            option_initial_price = BlackScholes.put_price(S0, K, T0, r, float(self.params.dividend_yield), float(self.params.sell_vol))
        else:
            option_initial_price = BlackScholes.call_price(S0, K, T0, r, float(self.params.dividend_yield), float(self.params.sell_vol))
        cash += (-contracts) * multiplier * option_initial_price  # contracts negative => inflow

        # Initial hedge using day's greek_vol
        delta0 = (
            BlackScholes.put_delta(S0, K, T0, r, float(self.params.dividend_yield), vol0)
            if self.params.option_type.lower() == "put"
            else BlackScholes.call_delta(S0, K, T0, r, float(self.params.dividend_yield), vol0)
        )
        target_shares = int(round(-contracts * multiplier * delta0))
        cash -= target_shares * S0
        stock_holding = float(target_shares)

        # Set initial hedge bands
        if self.params.recenter_bands_daily:
            # For daily recentering, initialize using day 0 close and chosen band vol
            init_band_vol = vol0 if self.params.use_greek_vol_for_bands else float(self.params.hedging_vol)
            self._reset_bands(S0, init_band_vol)
        else:
            bands_vol0 = vol0 if self.params.use_greek_vol_for_bands else float(self.params.hedging_vol)
            self._reset_bands(S0, bands_vol0)

        # Mark-to-market at end of first day
        option_price_0 = (
            BlackScholes.put_price(S0, K, T0, r, float(self.params.dividend_yield), vol0)
            if self.params.option_type.lower() == "put"
            else BlackScholes.call_price(S0, K, T0, r, float(self.params.dividend_yield), vol0)
        )
        delta_0 = (
            BlackScholes.put_delta(S0, K, T0, r, float(self.params.dividend_yield), vol0)
            if self.params.option_type.lower() == "put"
            else BlackScholes.call_delta(S0, K, T0, r, float(self.params.dividend_yield), vol0)
        )
        gamma_0 = BlackScholes.call_gamma(S0, K, T0, r, float(self.params.dividend_yield), vol0)
        theta_0_day = (
            BlackScholes.put_theta(S0, K, T0, r, float(self.params.dividend_yield), vol0)
            if self.params.option_type.lower() == "put"
            else BlackScholes.call_theta(S0, K, T0, r, float(self.params.dividend_yield), vol0)
        ) / float(self.params.trading_days_per_year)
        vega_0 = BlackScholes.call_vega(S0, K, T0, r, float(self.params.dividend_yield), vol0)
        option_value = contracts * multiplier * option_price_0
        stock_value = stock_holding * S0
        portfolio_value = cash + stock_value + option_value

        self.portfolio.loc[first_date, [
            "cash",
            "stock_holding",
            "stock_value",
            "option_value",
            "portfolio_value",
            "pnl",
            "option_delta",
            "option_gamma",
            "option_theta_per_day",
            "option_vega",
            "theta_pnl_day",
            "gamma_pnl_day",
            "vega_pnl_day",
            "financing_pnl_day",
            "residual_pnl",
        ]] = [cash, stock_holding, stock_value, option_value, portfolio_value, portfolio_value, delta_0, gamma_0, theta_0_day, vega_0, theta_0_day * contracts * multiplier, 0.0, 0.0, 0.0, 0.0]

        prev_portfolio_value = portfolio_value

        # -----------------------------
        # Step 5: Daily Backtesting Loop
        # -----------------------------
        idx = 1
        while idx < len(self.data.index):
            current_date = self.data.index[idx]
            S = float(self.data.loc[current_date, "Close"].iloc[0] if isinstance(self.data.loc[current_date, "Close"], pd.Series) else self.data.loc[current_date, "Close"]) 
            vol = float(self.data.loc[current_date, "greek_vol"].iloc[0] if isinstance(self.data.loc[current_date, "greek_vol"], pd.Series) else self.data.loc[current_date, "greek_vol"]) 
            T = self._year_fraction_to_expiry(current_date)

            # Apply daily financing on opening cash balance
            opening_cash = cash
            dt = 1.0 / 365.0
            if opening_cash >= 0.0:
                cash += opening_cash * float(self.params.cash_rate) * dt
            else:
                cash += opening_cash * float(self.params.borrow_rate) * dt
            interest_delta = cash - opening_cash  # freeze pure interest before any trading

            # Optionally recenter bands daily based on yesterday's close using hedging_vol
            if self.params.recenter_bands_daily and idx > 0:
                # Recenter bands daily using yesterday's close and band width source
                prev_close = float(self.data["Close"].to_numpy()[idx - 1])  # S(t-1)
                prev_vol = float(self.data["greek_vol"].to_numpy()[idx - 1])
                band_vol_daily = prev_vol if self.params.use_greek_vol_for_bands else float(self.params.hedging_vol)
                self._reset_bands(prev_close, band_vol_daily)

            # Check hedge trigger against today's close S(t)
            rehedged = False
            if self.upper_band is not None and self.lower_band is not None:
                if S >= self.upper_band or S <= self.lower_band:
                    # Re-hedge to new delta target
                    delta = (
                        BlackScholes.put_delta(S, K, T, r, float(self.params.dividend_yield), vol)
                        if self.params.option_type.lower() == "put"
                        else BlackScholes.call_delta(S, K, T, r, float(self.params.dividend_yield), vol)
                    )
                    target = int(round(-contracts * multiplier * delta))
                    diff = target - stock_holding
                    if diff != 0:
                        cash -= diff * S
                        stock_holding = float(target)
                    if not self.params.recenter_bands_daily:
                        # Original behavior: recenter upon hedge
                        bands_vol = vol if self.params.use_greek_vol_for_bands else float(self.params.hedging_vol)
                        self._reset_bands(S, bands_vol)
                    rehedged = True

            # Mark-to-market option value using day's greek_vol
            option_price = (
                BlackScholes.put_price(S, K, T, r, float(self.params.dividend_yield), vol)
                if self.params.option_type.lower() == "put"
                else BlackScholes.call_price(S, K, T, r, float(self.params.dividend_yield), vol)
            )
            delta_today = (
                BlackScholes.put_delta(S, K, T, r, float(self.params.dividend_yield), vol)
                if self.params.option_type.lower() == "put"
                else BlackScholes.call_delta(S, K, T, r, float(self.params.dividend_yield), vol)
            )
            gamma_today = BlackScholes.call_gamma(S, K, T, r, float(self.params.dividend_yield), vol)
            theta_today_day = (
                BlackScholes.put_theta(S, K, T, r, float(self.params.dividend_yield), vol)
                if self.params.option_type.lower() == "put"
                else BlackScholes.call_theta(S, K, T, r, float(self.params.dividend_yield), vol)
            ) / float(self.params.trading_days_per_year)
            vega_today = BlackScholes.call_vega(S, K, T, r, float(self.params.dividend_yield), vol)
            option_value = contracts * multiplier * option_price
            stock_value = stock_holding * S

            # Portfolio valuation and PnL
            portfolio_value = cash + stock_value + option_value
            pnl = portfolio_value - prev_portfolio_value

            # PnL attribution using close-to-close squared move and theta per day
            prev_date = self.data.index[idx - 1]
            S_prev = float(self.data.loc[prev_date, "Close"].iloc[0] if isinstance(self.data.loc[prev_date, "Close"], pd.Series) else self.data.loc[prev_date, "Close"]) 
            dS = S - S_prev
            gamma_pnl_day = 0.5 * gamma_today * (dS ** 2) * contracts * multiplier
            theta_pnl_day = theta_today_day * contracts * multiplier
            # Vega attribution uses change in the vol used for Greeks/MTM (greek_vol)
            vol_prev = float(self.data.loc[prev_date, "greek_vol"].iloc[0] if isinstance(self.data.loc[prev_date, "greek_vol"], pd.Series) else self.data.loc[prev_date, "greek_vol"]) 
            dSigma = vol - vol_prev
            vega_pnl_day = vega_today * dSigma * contracts * multiplier
            financing_pnl_day = interest_delta
            residual_pnl = pnl - gamma_pnl_day - theta_pnl_day - vega_pnl_day - financing_pnl_day

            self.portfolio.loc[current_date, [
                "cash",
                "stock_holding",
                "stock_value",
                "option_value",
                "portfolio_value",
                "pnl",
                "option_delta",
                "option_gamma",
                "option_theta_per_day",
                "option_vega",
                "theta_pnl_day",
                "gamma_pnl_day",
                "vega_pnl_day",
                "financing_pnl_day",
                "residual_pnl",
            ]] = [cash, stock_holding, stock_value, option_value, portfolio_value, pnl, delta_today, gamma_today, theta_today_day, vega_today, theta_pnl_day, gamma_pnl_day, vega_pnl_day, financing_pnl_day, residual_pnl]

            prev_portfolio_value = portfolio_value
            idx += 1

        # -----------------------------------------
        # Step 6: Final Settlement on the last day
        # -----------------------------------------
        last_date = self.data.index[-1]
        final_price = float(self.data.loc[last_date, "Close"].iloc[0] if isinstance(self.data.loc[last_date, "Close"], pd.Series) else self.data.loc[last_date, "Close"]) 
        T_last = self._year_fraction_to_expiry(last_date)

        # Close the hedge
        cash += stock_holding * final_price
        stock_holding = 0.0

        # If expired or last date is the expiry, settle intrinsic; otherwise, close option at model value
        if pd.to_datetime(last_date).normalize() >= pd.to_datetime(self.params.expiry_date).normalize() or T_last == 0.0:
            if self.params.option_type.lower() == "put":
                call_settlement = max(0.0, K - final_price)
            else:
                call_settlement = max(0.0, final_price - K)
        else:
            # Close option at greek_vol MTM
            vol_last = float(self.data.loc[last_date, "greek_vol"].iloc[0] if isinstance(self.data.loc[last_date, "greek_vol"], pd.Series) else self.data.loc[last_date, "greek_vol"]) 
            call_settlement = (
                BlackScholes.put_price(final_price, K, T_last, r, float(self.params.dividend_yield), vol_last)
                if self.params.option_type.lower() == "put"
                else BlackScholes.call_price(final_price, K, T_last, r, float(self.params.dividend_yield), vol_last)
            )

        # For a short position (contracts negative), buying back costs cash
        cash += contracts * multiplier * call_settlement

        # Final portfolio value and final-day PnL adjustment
        final_portfolio_value = cash
        final_pnl = final_portfolio_value - prev_portfolio_value

        self.portfolio.loc[last_date, [
            "cash",
            "stock_holding",
            "stock_value",
            "option_value",
            "portfolio_value",
            "pnl",
        ]] = [cash, 0.0, 0.0, 0.0, final_portfolio_value, final_pnl]

        # Step 7: Add cumulative PnL
        self.portfolio["cumulative_pnl"] = self.portfolio["pnl"].cumsum()

        return self.portfolio

    def plot_results(self) -> None:
        # Plot cumulative PnL vs underlying price
        fig, ax_left = plt.subplots(figsize=(12, 6))
        ax_right = ax_left.twinx()

        ax_left.plot(self.portfolio.index, self.portfolio["cumulative_pnl"], color="tab:blue", label="Cumulative PnL")
        ax_right.plot(self.data.index, self.data["Close"], color="tab:orange", label=f"{self.params.ticker} Close")
        # Strike level (price axis)
        ax_right.axhline(y=float(self.params.strike_price), color="gray", linestyle=":", linewidth=1.5, label="Strike")

        ax_left.set_xlabel("Date")
        ax_left.set_ylabel("Cumulative PnL", color="tab:blue")
        ax_right.set_ylabel("Price", color="tab:orange")
        ax_left.grid(True, linestyle="--", alpha=0.3)

        title = (
            f"Delta Hedge Backtest | {self.params.ticker} | Strike {self.params.strike_price} | "
            f"{pd.to_datetime(self.params.trade_date).date()} to {pd.to_datetime(self.params.expiry_date).date()}"
        )
        plt.title(title)

        # Build a combined legend
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")
        plt.tight_layout()
        plt.show()

    def plot_attribution(self) -> None:
        # Layered filled areas (not stacked) to avoid color overlap ambiguity with negatives
        cum_theta = self.portfolio["theta_pnl_day"].cumsum()
        cum_gamma = self.portfolio["gamma_pnl_day"].cumsum()
        cum_vega = self.portfolio["vega_pnl_day"].cumsum()
        cum_resid = self.portfolio["residual_pnl"].cumsum()
        cum_total = self.portfolio["cumulative_pnl"]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = self.portfolio.index

        # Draw residual and vega first, then gamma, then theta on top for visibility
        resid_area = ax.fill_between(x, 0, cum_resid, color="#2196F3", alpha=0.30, label="Residual", zorder=1)
        vega_area = ax.fill_between(x, 0, cum_vega, color="#9C27B0", alpha=0.40, label="Vega", zorder=2)
        gamma_area = ax.fill_between(x, 0, cum_gamma, color="#F44336", alpha=0.45, label="Gamma", zorder=3)
        theta_area = ax.fill_between(x, 0, cum_theta, color="#4CAF50", alpha=0.50, label="Theta", zorder=4)
        total_line, = ax.plot(x, cum_total, color="black", linewidth=2.0, label="Total", zorder=5)

        ax.set_title("Cumulative PnL Attribution")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(handles=[theta_area, gamma_area, vega_area, resid_area, total_line],
                  labels=["Theta", "Gamma", "Vega", "Residual", "Total"],
                  loc="best")
        plt.tight_layout()
        plt.show()

    def plot_financing(self) -> None:
        # Financing attribution vs total
        cum_fin = self.portfolio["financing_pnl_day"].cumsum()
        cum_total = self.portfolio["cumulative_pnl"]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.portfolio.index, cum_total, color="black", linewidth=2.0, label="Total")
        ax.plot(self.portfolio.index, cum_fin, color="#795548", linewidth=2.0, label="Financing")
        ax.set_title("Cumulative Financing vs Total")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_realized_vol(self) -> None:
        # 90d annualized realized vol series used for Greeks
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.data.index, self.data["greek_vol"], color="tab:purple", label=f"{self.params.greek_vol_lookback}d realized vol (annualized)")
        ax.set_title(f"{self.params.ticker} {self.params.greek_vol_lookback}d Realized Volatility (annualized)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Realized Vol")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()


def main() -> None:
    # Example: fixed hedging window, expiry aligned to hedging_end
    hedge_start = pd.to_datetime("2025-03-01")
    hedge_end = pd.to_datetime("2025-08-31")
    trade_dt = hedge_start
    # Align option expiry with hedging end date to match the hedging window
    expiry_dt = hedge_end

    params = StrategyParams(
        ticker="SPY",
        strike_price=500,
        expiry_date=expiry_dt,
        trade_date=trade_dt,
        risk_free_rate=0.05,
        option_contracts=-1,  # sell 1 contract
        sell_vol=0.40,
        hedging_vol=0.30,
        greek_vol_lookback=90,
        hedging_start_date=hedge_start,
        hedging_end_date=hedge_end,
        strict_window=True,  # enforce exact hedging window
        option_type="Call",
    )

    # Fetch data and prepare features
    fetcher = MarketDataFetcher(params)
    data = fetcher.fetch_and_prepare()

    # Run backtest
    backtester = DeltaHedgeBacktester(params, data)
    portfolio = backtester.run()

    # Plot
    backtester.plot_results()
    backtester.plot_attribution()
    backtester.plot_financing()
    backtester.plot_realized_vol()

    # Comparison: same sell_vol, change hedging_vol to 0.40
    params_cmp = StrategyParams(
        ticker=params.ticker,
        strike_price=params.strike_price,
        expiry_date=params.expiry_date,
        trade_date=params.trade_date,
        risk_free_rate=params.risk_free_rate,
        option_contracts=params.option_contracts,
        sell_vol=params.sell_vol,
        hedging_vol=0.40,
        greek_vol_lookback=params.greek_vol_lookback,
        hedging_start_date=params.hedging_start_date,
        hedging_end_date=params.hedging_end_date,
        strict_window=params.strict_window,
    )
    backtester_cmp = DeltaHedgeBacktester(params_cmp, data)
    portfolio_cmp = backtester_cmp.run()

    # Third scenario: use realized greek_vol for bands (dynamic bands)
    params_dyn = StrategyParams(
        ticker=params.ticker,
        strike_price=params.strike_price,
        expiry_date=params.expiry_date,
        trade_date=params.trade_date,
        risk_free_rate=params.risk_free_rate,
        option_contracts=params.option_contracts,
        sell_vol=params.sell_vol,
        hedging_vol=params.hedging_vol,  # fallback, unused when flag True
        greek_vol_lookback=params.greek_vol_lookback,
        hedging_start_date=params.hedging_start_date,
        hedging_end_date=params.hedging_end_date,
        strict_window=params.strict_window,
        use_greek_vol_for_bands=True,
    )
    backtester_dyn = DeltaHedgeBacktester(params_dyn, data)
    portfolio_dyn = backtester_dyn.run()

    # Plot comparison of cumulative PnL
    fig, ax_left = plt.subplots(figsize=(12, 6))
    ax_right = ax_left.twinx()
    ax_left.plot(portfolio.index, portfolio["cumulative_pnl"], label=f"Hedge vol {params.hedging_vol:.2f}", color="tab:blue")
    ax_left.plot(portfolio_cmp.index, portfolio_cmp["cumulative_pnl"], label=f"Hedge vol {params_cmp.hedging_vol:.2f}", color="tab:green")
    ax_left.plot(portfolio_dyn.index, portfolio_dyn["cumulative_pnl"], label=f"Bands = {params.greek_vol_lookback}d realized vol", color="tab:red")
    ax_right.plot(data.index, data["Close"], color="tab:orange", alpha=0.6, label=f"{params.ticker} Close")

    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Cumulative PnL")
    ax_right.set_ylabel("Price")
    ax_left.grid(True, linestyle="--", alpha=0.3)
    plt.title(f"Cumulative PnL Comparison: Hedge vol {params.hedging_vol:.2f} vs {params_cmp.hedging_vol:.2f} vs bands={params.greek_vol_lookback}d RV (sell vol {params.sell_vol:.2f})")

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")
    plt.tight_layout()
    plt.show()

    # Ex-financing overlay and final comparison
    def _cum_ex_fin(df: pd.DataFrame) -> pd.Series:
        fin = df.get("financing_pnl_day", pd.Series(0.0, index=df.index)).cumsum()
        return df["cumulative_pnl"] - fin

    base_ex = _cum_ex_fin(portfolio)
    cmp_ex = _cum_ex_fin(portfolio_cmp)
    dyn_ex = _cum_ex_fin(portfolio_dyn)

    fig_ex, ax_ex = plt.subplots(figsize=(12, 4))
    ax_ex.plot(portfolio.index, portfolio["cumulative_pnl"], color="tab:blue", label="Base with fin")
    ax_ex.plot(portfolio.index, base_ex, color="tab:blue", linestyle="--", label="Base ex-fin")
    ax_ex.plot(portfolio_cmp.index, portfolio_cmp["cumulative_pnl"], color="tab:green", label="Cmp with fin")
    ax_ex.plot(portfolio_cmp.index, cmp_ex, color="tab:green", linestyle="--", label="Cmp ex-fin")
    ax_ex.plot(portfolio_dyn.index, portfolio_dyn["cumulative_pnl"], color="tab:red", label="RV with fin")
    ax_ex.plot(portfolio_dyn.index, dyn_ex, color="tab:red", linestyle="--", label="RV ex-fin")
    ax_ex.set_title("Cumulative PnL: with vs ex-financing (overlay)")
    ax_ex.set_xlabel("Date")
    ax_ex.set_ylabel("Cumulative PnL")
    ax_ex.grid(True, linestyle="--", alpha=0.3)
    ax_ex.legend(loc="best", ncol=3)
    plt.tight_layout()
    plt.show()

    labels = ["Base", "Compare", "Bands=RV"]
    with_fin = [
        float(portfolio["cumulative_pnl"].iloc[-1]),
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
    plt.tight_layout()
    plt.show()

    # Optionally, print final stats
    final_value = float(portfolio["portfolio_value"].iloc[-1])
    print(f"Final portfolio value: {final_value:,.2f}")
    base_cum = float(portfolio["cumulative_pnl"].iloc[-1])
    cmp_cum = float(portfolio_cmp["cumulative_pnl"].iloc[-1])
    dyn_cum = float(portfolio_dyn["cumulative_pnl"].iloc[-1])
    diff_cum = base_cum - cmp_cum
    print(f"Final cumulative PnL (hedge vol {params.hedging_vol:.2f}): {base_cum:,.2f}")
    print(f"Final cumulative PnL (hedge vol {params_cmp.hedging_vol:.2f}): {cmp_cum:,.2f}")
    print(f"Difference (0.32 - 0.40): {diff_cum:,.2f}")
    print(f"Final cumulative PnL (bands = {params.greek_vol_lookback}d RV): {dyn_cum:,.2f}")
    # Ex-finance stats
    print("--- Ex-financing comparison ---")
    print(f"Base ex-fin: {base_ex.iloc[-1]:,.2f} | Δfin: {(with_fin[0]-ex_fin[0]):,.2f}")
    print(f"Compare ex-fin: {cmp_ex.iloc[-1]:,.2f} | Δfin: {(with_fin[1]-ex_fin[1]):,.2f}")
    print(f"Bands=RV ex-fin: {dyn_ex.iloc[-1]:,.2f} | Δfin: {(with_fin[2]-ex_fin[2]):,.2f}")

    # Plot re-hedge counts across the three scenarios
    def _count_rehedges(df: pd.DataFrame) -> int:
        return int((df["stock_holding"].diff().fillna(0) != 0).sum())

    counts = {
        f"Hedge vol {params.hedging_vol:.2f}": _count_rehedges(portfolio),
        f"Hedge vol {params_cmp.hedging_vol:.2f}": _count_rehedges(portfolio_cmp),
        f"Bands {params.greek_vol_lookback}d RV": _count_rehedges(portfolio_dyn),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color=["tab:blue", "tab:green", "tab:red"]) 
    ax.set_title("Number of Delta Re-hedges (by scenario)")
    ax.set_ylabel("Count of re-hedge events")
    for i, v in enumerate(counts.values()):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

    # Export results to Excel with multiple sheets for verification
    try:
        start_str = pd.to_datetime(params.trade_date).strftime('%Y%m%d')
        end_str = pd.to_datetime(params.expiry_date).strftime('%Y%m%d')
        filename = f"backtest_{params.ticker}_{start_str}_{end_str}.xlsx"
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            portfolio.to_excel(writer, sheet_name="portfolio_base")
            portfolio_cmp.to_excel(writer, sheet_name="portfolio_cmp")
            portfolio_dyn.to_excel(writer, sheet_name="portfolio_dyn")
            data.to_excel(writer, sheet_name="market_data")
        print(f"Exported Excel: {filename}")
    except Exception as exc:
        print(f"Excel export failed: {exc}")


if __name__ == "__main__":
    main()


