from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib as mpl


@dataclass
class MCParams:
    ticker: str = "SPY"
    S0: float = 500.0
    strike_price: float = 500.0
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0  # set to >0 if needed
    option_contracts: int = -1  # short 1
    sell_vol: float = 0.32  # vol to sell initial option
    hedging_vol: float = 0.32  # fallback band width vol
    path_vol: float = 0.32  # realized vol used to simulate paths
    greeks_vol_lookback: int = 60  # days for rolling realized vol used for Greeks/MTM
    T_years: float = 0.5  # 6 months
    steps_per_year: int = 252
    n_paths: int = 2000
    seed: int | None = 42
    contract_multiplier: int = 100
    use_greek_vol_for_bands: bool = False  # if True, use dynamic greeks vol for bands
    rehedge_time_step: int = 1  # hedge checks every step
    # Financing
    cash_rate: float = 0.0  # annual rate earned on positive cash
    borrow_rate: float = 0.035  # annual rate paid on negative cash
    # Option type: "Call" or "Put"
    option_type: str = "Call"


class BlackScholes:
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0:
            return max(S - K, 0.0)
        sigma = max(1e-10, sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def call_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0:
            return 1.0 if S > K else 0.0
        sigma = max(1e-10, sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        return math.exp(-q * T) * norm.cdf(d1)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0:
            return max(K - S, 0.0)
        sigma = max(1e-10, sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        if T <= 0:
            return -1.0 if S < K else 0.0
        sigma = max(1e-10, sigma)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        return math.exp(-q * T) * (norm.cdf(d1) - 1.0)


class MCDeltaHedgeSimulator:
    def __init__(self, params: MCParams):
        self.p = params
        if self.p.seed is not None:
            np.random.seed(self.p.seed)

    def _simulate_paths(self) -> np.ndarray:
        steps = int(self.p.steps_per_year * self.p.T_years)
        dt = 1.0 / self.p.steps_per_year
        mu = self.p.risk_free_rate - self.p.dividend_yield
        sigma = self.p.path_vol
        S = np.empty((self.p.n_paths, steps + 1), dtype=float)
        S[:, 0] = self.p.S0
        # Log-Euler GBM
        for t in range(1, steps + 1):
            z = np.random.normal(size=self.p.n_paths)
            S[:, t] = S[:, t - 1] * np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)
        return S

    def _daily_band_width(self, vol: float) -> float:
        return vol / math.sqrt(self.p.steps_per_year)

    def run(self) -> pd.DataFrame:
        S_paths = self._simulate_paths()
        steps = S_paths.shape[1] - 1
        dt = 1.0 / self.p.steps_per_year

        # Results per path
        final_values = np.zeros(self.p.n_paths, dtype=float)
        final_values_ex_fin = np.zeros(self.p.n_paths, dtype=float)

        for i in range(self.p.n_paths):
            # Price path and dynamic Greeks vol from rolling realized vol
            prices = pd.Series(S_paths[i, :])
            log_ret = np.log(prices).diff()
            lookback = int(self.p.greeks_vol_lookback)
            rolling_std = log_ret.rolling(window=lookback, min_periods=lookback).std()
            greeks_vol = (rolling_std * math.sqrt(self.p.steps_per_year)).bfill().ffill().fillna(float(self.p.path_vol))

            S0 = float(prices.iloc[0])
            cash = 0.0
            stock = 0.0

            # Sell/Buy option using sell_vol
            if self.p.option_type.lower() == "put":
                price0 = BlackScholes.put_price(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, self.p.sell_vol)
            else:
                price0 = BlackScholes.call_price(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, self.p.sell_vol)
            cash += (-self.p.option_contracts) * self.p.contract_multiplier * price0

            # Initial hedge using dynamic Greeks vol at t=0
            vol0 = float(greeks_vol.iloc[0])
            delta0 = (
                BlackScholes.put_delta(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, vol0)
                if self.p.option_type.lower() == "put"
                else BlackScholes.call_delta(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, vol0)
            )
            target = int(round(-self.p.option_contracts * self.p.contract_multiplier * delta0))
            cash -= target * S0
            stock = float(target)

            # Bands
            band_vol0 = vol0 if self.p.use_greek_vol_for_bands else self.p.hedging_vol
            daily_move_pct = self._daily_band_width(band_vol0)
            upper = S0 * (1.0 + daily_move_pct)
            lower = S0 * (1.0 - daily_move_pct)

            # Evolve over steps
            cum_financing = 0.0
            for t in range(1, steps + 1):
                remaining_T = max(0.0, self.p.T_years - t * dt)
                S = float(prices.iloc[t])
                vol_t = float(greeks_vol.iloc[t])

                # Financing on opening cash (calendar 365 day convention)
                opening_cash = cash
                dt_cal = 1.0 / 365.0
                if opening_cash >= 0.0:
                    interest = opening_cash * float(self.p.cash_rate) * dt_cal
                else:
                    interest = opening_cash * float(self.p.borrow_rate) * dt_cal
                cash += interest
                cum_financing += interest

                # Hedge check
                if (t % self.p.rehedge_time_step == 0) and (S >= upper or S <= lower):
                    delta = (
                        BlackScholes.put_delta(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                        if self.p.option_type.lower() == "put"
                        else BlackScholes.call_delta(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                    )
                    target = int(round(-self.p.option_contracts * self.p.contract_multiplier * delta))
                    diff = target - stock
                    if diff != 0:
                        cash -= diff * S
                        stock = float(target)
                    band_vol = vol_t if self.p.use_greek_vol_for_bands else self.p.hedging_vol
                    daily_move_pct = self._daily_band_width(band_vol)
                    upper = S * (1.0 + daily_move_pct)
                    lower = S * (1.0 - daily_move_pct)

                # End step MTM (no need to store per-step results per path)
                option_price = (
                    BlackScholes.put_price(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                    if self.p.option_type.lower() == "put"
                    else BlackScholes.call_price(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                )
                option_value = self.p.option_contracts * self.p.contract_multiplier * option_price
                portfolio_value = cash + stock * S + option_value

            # Final settlement at T
            S_final = float(prices.iloc[-1])
            cash += stock * S_final
            stock = 0.0
            payoff = (
                max(0.0, self.p.strike_price - S_final)
                if self.p.option_type.lower() == "put"
                else max(0.0, S_final - self.p.strike_price)
            )
            cash += self.p.option_contracts * self.p.contract_multiplier * payoff
            final_values[i] = cash
            final_values_ex_fin[i] = cash - cum_financing

        df = pd.DataFrame({
            "final_portfolio_value": final_values,
            "final_portfolio_value_ex_fin": final_values_ex_fin,
        })
        return df

    def _select_sample_paths(self, finals: pd.DataFrame, quantiles: list[float]) -> list[int]:
        vals = finals["final_portfolio_value"].values
        path_ids = list(range(len(vals)))
        targets = list(np.quantile(vals, quantiles))
        # also include min, mean, max
        targets.extend([float(np.min(vals)), float(np.mean(vals)), float(np.max(vals))])
        chosen: list[int] = []
        used = set()
        for tgt in targets:
            diffs = np.abs(vals - tgt)
            order = np.argsort(diffs)
            chosen_id = None
            for idx in order:
                if idx not in used:
                    chosen_id = int(idx)
                    break
            if chosen_id is not None:
                chosen.append(chosen_id)
                used.add(chosen_id)
        return chosen

    def _replay_path_timeseries(self, path_id: int, prices: np.ndarray) -> pd.DataFrame:
        steps = len(prices) - 1
        dt = 1.0 / self.p.steps_per_year
        # Greeks vol from rolling realized vol
        series = pd.Series(prices)
        log_ret = np.log(series).diff()
        lookback = int(self.p.greeks_vol_lookback)
        rolling_std = log_ret.rolling(window=lookback, min_periods=lookback).std()
        greeks_vol = (rolling_std * math.sqrt(self.p.steps_per_year)).bfill().ffill().fillna(float(self.p.path_vol))

        S0 = float(series.iloc[0])
        cash = 0.0
        stock = 0.0
        cum_fin = 0.0

        # Initial option premium
        if self.p.option_type.lower() == "put":
            price0 = BlackScholes.put_price(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, self.p.sell_vol)
        else:
            price0 = BlackScholes.call_price(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, self.p.sell_vol)
        cash += (-self.p.option_contracts) * self.p.contract_multiplier * price0

        vol0 = float(greeks_vol.iloc[0])
        delta0 = (
            BlackScholes.put_delta(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, vol0)
            if self.p.option_type.lower() == "put"
            else BlackScholes.call_delta(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, vol0)
        )
        target = int(round(-self.p.option_contracts * self.p.contract_multiplier * delta0))
        cash -= target * S0
        stock = float(target)

        # Bands
        band_vol0 = vol0 if self.p.use_greek_vol_for_bands else self.p.hedging_vol
        daily_move_pct = band_vol0 / math.sqrt(self.p.steps_per_year)
        upper = S0 * (1.0 + daily_move_pct)
        lower = S0 * (1.0 - daily_move_pct)

        rows = []
        for t in range(1, steps + 1):
            remaining_T = max(0.0, self.p.T_years - t * dt)
            S = float(series.iloc[t])
            vol_t = float(greeks_vol.iloc[t])

            opening_cash = cash
            dt_cal = 1.0 / 365.0
            interest = opening_cash * (self.p.cash_rate if opening_cash >= 0.0 else self.p.borrow_rate) * dt_cal
            cash += interest
            cum_fin += interest

            hedge_triggered = False
            if (t % self.p.rehedge_time_step == 0) and (S >= upper or S <= lower):
                delta = (
                    BlackScholes.put_delta(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                    if self.p.option_type.lower() == "put"
                    else BlackScholes.call_delta(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                )
                target2 = int(round(-self.p.option_contracts * self.p.contract_multiplier * delta))
                diff = target2 - stock
                if diff != 0:
                    cash -= diff * S
                    stock = float(target2)
                    hedge_triggered = True
                band_vol = vol_t if self.p.use_greek_vol_for_bands else self.p.hedging_vol
                daily_move_pct = band_vol / math.sqrt(self.p.steps_per_year)
                upper = S * (1.0 + daily_move_pct)
                lower = S * (1.0 - daily_move_pct)

            option_price = (
                BlackScholes.put_price(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                if self.p.option_type.lower() == "put"
                else BlackScholes.call_price(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
            )
            option_value = self.p.option_contracts * self.p.contract_multiplier * option_price
            portfolio_value = cash + stock * S + option_value

            rows.append({
                "path_id": int(path_id),
                "step": int(t),
                "S": float(S),
                "stock": float(stock),
                "cash": float(cash),
                "opening_cash": float(opening_cash),
                "interest_day": float(interest),
                "cum_financing": float(cum_fin),
                "option_price": float(option_price),
                "portfolio_value": float(portfolio_value),
                "hedge_triggered": bool(hedge_triggered),
            })

        return pd.DataFrame(rows)

    def run_with_details(self, sample_quantiles: list[float] | None = None, return_events: bool = True) -> dict:
        """
        Run simulation and return artifacts for export.
        Returns dict with keys: finals (DataFrame), summary (DataFrame), events (DataFrame|None), samples (DataFrame|None)
        """
        S_paths = self._simulate_paths()
        steps = S_paths.shape[1] - 1
        dt = 1.0 / self.p.steps_per_year

        final_values = np.zeros(self.p.n_paths, dtype=float)
        final_values_ex_fin = np.zeros(self.p.n_paths, dtype=float)
        cum_fin_paths = np.zeros(self.p.n_paths, dtype=float)
        events_rows: list[dict] = []

        for i in range(self.p.n_paths):
            prices = pd.Series(S_paths[i, :])
            log_ret = np.log(prices).diff()
            lookback = int(self.p.greeks_vol_lookback)
            rolling_std = log_ret.rolling(window=lookback, min_periods=lookback).std()
            greeks_vol = (rolling_std * math.sqrt(self.p.steps_per_year)).bfill().ffill().fillna(float(self.p.path_vol))

            S0 = float(prices.iloc[0])
            cash = 0.0
            stock = 0.0

            if self.p.option_type.lower() == "put":
                price0 = BlackScholes.put_price(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, self.p.sell_vol)
            else:
                price0 = BlackScholes.call_price(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, self.p.sell_vol)
            cash += (-self.p.option_contracts) * self.p.contract_multiplier * price0

            vol0 = float(greeks_vol.iloc[0])
            delta0 = (
                BlackScholes.put_delta(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, vol0)
                if self.p.option_type.lower() == "put"
                else BlackScholes.call_delta(S0, self.p.strike_price, self.p.T_years, self.p.risk_free_rate, self.p.dividend_yield, vol0)
            )
            target = int(round(-self.p.option_contracts * self.p.contract_multiplier * delta0))
            cash -= target * S0
            stock = float(target)

            band_vol0 = vol0 if self.p.use_greek_vol_for_bands else self.p.hedging_vol
            daily_move_pct = band_vol0 / math.sqrt(self.p.steps_per_year)
            upper = S0 * (1.0 + daily_move_pct)
            lower = S0 * (1.0 - daily_move_pct)

            cum_financing = 0.0
            for t in range(1, steps + 1):
                remaining_T = max(0.0, self.p.T_years - t * dt)
                S = float(prices.iloc[t])
                vol_t = float(greeks_vol.iloc[t])

                opening_cash = cash
                dt_cal = 1.0 / 365.0
                interest = opening_cash * (self.p.cash_rate if opening_cash >= 0.0 else self.p.borrow_rate) * dt_cal
                cash += interest
                cum_financing += interest
                if return_events:
                    events_rows.append({
                        "path_id": int(i),
                        "step": int(t),
                        "event": "financing",
                        "S": float(S),
                        "opening_cash": float(opening_cash),
                        "interest": float(interest),
                        "cum_financing": float(cum_financing),
                    })

                if (t % self.p.rehedge_time_step == 0) and (S >= upper or S <= lower):
                    delta = (
                        BlackScholes.put_delta(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                        if self.p.option_type.lower() == "put"
                        else BlackScholes.call_delta(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                    )
                    target2 = int(round(-self.p.option_contracts * self.p.contract_multiplier * delta))
                    diff = target2 - stock
                    if diff != 0:
                        cash -= diff * S
                        if return_events:
                            events_rows.append({
                                "path_id": int(i),
                                "step": int(t),
                                "event": "hedge",
                                "S": float(S),
                                "trade_shares": float(diff),
                                "trade_cash": float(-diff * S),
                            })
                        stock = float(target2)
                    band_vol = vol_t if self.p.use_greek_vol_for_bands else self.p.hedging_vol
                    daily_move_pct = band_vol / math.sqrt(self.p.steps_per_year)
                    upper = S * (1.0 + daily_move_pct)
                    lower = S * (1.0 - daily_move_pct)

                option_price = (
                    BlackScholes.put_price(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                    if self.p.option_type.lower() == "put"
                    else BlackScholes.call_price(S, self.p.strike_price, remaining_T, self.p.risk_free_rate, self.p.dividend_yield, vol_t)
                )
                option_value = self.p.option_contracts * self.p.contract_multiplier * option_price
                portfolio_value = cash + stock * S + option_value
                if return_events:
                    events_rows.append({
                        "path_id": int(i),
                        "step": int(t),
                        "event": "eod",
                        "portfolio_value": float(portfolio_value),
                    })

            S_final = float(prices.iloc[-1])
            cash += stock * S_final
            stock = 0.0
            payoff = (
                max(0.0, self.p.strike_price - S_final)
                if self.p.option_type.lower() == "put"
                else max(0.0, S_final - self.p.strike_price)
            )
            cash += self.p.option_contracts * self.p.contract_multiplier * payoff
            final_values[i] = cash
            final_values_ex_fin[i] = cash - cum_financing
            cum_fin_paths[i] = cum_financing

        finals = pd.DataFrame({
            "path_id": np.arange(self.p.n_paths, dtype=int),
            "final_portfolio_value": final_values,
            "final_portfolio_value_ex_fin": final_values_ex_fin,
            "cum_financing": cum_fin_paths,
        })

        # Summary
        with_vals = finals["final_portfolio_value"].values
        summary = pd.DataFrame({
            "metric": ["mean", "p5", "p50", "p95"],
            "with_fin": [float(np.mean(with_vals))] + list(np.percentile(with_vals, [5, 50, 95])),
        })
        if "final_portfolio_value_ex_fin" in finals.columns:
            ex_vals = finals["final_portfolio_value_ex_fin"].values
            summary["ex_fin"] = [float(np.mean(ex_vals))] + list(np.percentile(ex_vals, [5, 50, 95]))

        # Samples
        samples_df = None
        if sample_quantiles is not None and len(sample_quantiles) > 0:
            sample_ids = self._select_sample_paths(finals, sample_quantiles)
            sample_rows = []
            for pid in sample_ids:
                df_path = self._replay_path_timeseries(pid, S_paths[pid, :])
                df_path["label"] = "sample"
                sample_rows.append(df_path)
            if sample_rows:
                samples_df = pd.concat(sample_rows, ignore_index=True)

        events_df = pd.DataFrame(events_rows) if return_events and len(events_rows) > 0 else None

        return {"finals": finals, "summary": summary, "events": events_df, "samples": samples_df}

    def plot_distribution(self, df: pd.DataFrame) -> None:
        vals = df["final_portfolio_value"].values
        mean = float(np.mean(vals))
        p5, p50, p95 = np.percentile(vals, [5, 50, 95])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(vals, bins=50, color="tab:blue", alpha=0.6, edgecolor="white")
        ax.axvline(mean, color="black", linestyle="--", label=f"Mean: {mean:,.0f}")
        ax.axvline(p50, color="tab:green", linestyle=":", label=f"Median: {p50:,.0f}")
        ax.axvspan(p5, p95, color="tab:orange", alpha=0.2, label=f"5th–95th: [{p5:,.0f}, {p95:,.0f}]")
        ax.set_title("Monte Carlo Final Portfolio Value Distribution")
        ax.set_xlabel("Final Portfolio Value")
        ax.set_ylabel("Frequency")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_distribution_dual(self, df: pd.DataFrame) -> None:
        """Overlay histograms for with-financing vs ex-financing results."""
        with_fin = df["final_portfolio_value"].values
        ex_fin = df.get("final_portfolio_value_ex_fin", pd.Series(np.nan, index=df.index)).values

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(with_fin, bins=50, color="tab:blue", alpha=0.45, edgecolor="white", label="With financing")
        if np.isfinite(ex_fin).all():
            ax.hist(ex_fin, bins=50, color="tab:green", alpha=0.45, edgecolor="white", label="Ex-financing")
        ax.set_title("Monte Carlo Final Portfolio Value: with vs ex-financing")
        ax.set_xlabel("Final Portfolio Value")
        ax.set_ylabel("Frequency")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()


def sweep_sell_vols(base_params: MCParams, sell_vols: list[float]) -> pd.DataFrame:
    """
    Run multiple scenarios varying sell_vol while holding path_vol (realized) fixed.

    Returns a DataFrame with columns: sell_vol, dv (sell_vol - mtm_vol),
    mean, p5, p50, p95 of final portfolio value.
    """
    results = []
    for sv in sell_vols:
        scenario_params = MCParams(
            ticker=base_params.ticker,
            S0=base_params.S0,
            strike_price=base_params.strike_price,
            risk_free_rate=base_params.risk_free_rate,
            dividend_yield=base_params.dividend_yield,
            option_contracts=base_params.option_contracts,
            sell_vol=sv,
            hedging_vol=base_params.hedging_vol,
            path_vol=base_params.path_vol,
            greeks_vol_lookback=base_params.greeks_vol_lookback,
            T_years=base_params.T_years,
            steps_per_year=base_params.steps_per_year,
            n_paths=base_params.n_paths,
            seed=base_params.seed,
            contract_multiplier=base_params.contract_multiplier,
            use_greek_vol_for_bands=base_params.use_greek_vol_for_bands,
            cash_rate=base_params.cash_rate,
            borrow_rate=base_params.borrow_rate,
        )
        sim = MCDeltaHedgeSimulator(scenario_params)
        df = sim.run()
        vals = df["final_portfolio_value"].values
        mean = float(np.mean(vals))
        p5, p50, p95 = np.percentile(vals, [5, 50, 95])
        results.append({
            "sell_vol": sv,
            "dv": sv - base_params.path_vol,
            "mean": mean,
            "p5": p5,
            "p50": p50,
            "p95": p95,
        })
    out = pd.DataFrame(results).sort_values("sell_vol").reset_index(drop=True)
    return out


def plot_sweep_heatmap(sweep_df: pd.DataFrame) -> None:
    """Plot a modern heatmap of key metrics vs sell_vol using a color gradient."""
    # Prepare a neat table with rounded values for display
    display = sweep_df.copy()
    # Round dollar columns to nearest dollar for readability
    for col in ["mean", "p5", "p50", "p95"]:
        display[col] = display[col].round(0)

    # Build a matrix form where columns are metrics and rows are sell_vol
    metrics = ["mean", "p5", "p50", "p95"]
    mat = display.set_index("sell_vol")[metrics]
    # Map of sell_vol -> dv for axis labels
    dv_map = dict(zip(display["sell_vol"], display["dv"]))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    cmap = mpl.cm.get_cmap("RdYlGn")
    # Normalize around zero so reds are losses, greens profits
    vmax = float(np.nanmax(np.abs(mat.values)))
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)

    # Annotate with values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            ax.text(j, i, f"{val:,.0f}", ha="center", va="center", color="black", fontsize=9)

    # Axes labels and ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([f"{sv:.2f} (dv {dv_map.get(sv, float('nan')):+.2f})" for sv in mat.index])
    ax.set_xlabel("Metric")
    ax.set_ylabel("sell_vol (with dv)")
    ax.set_title("Sell vol sweep vs fixed path_vol — heatmap ($)")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("$", rotation=270, labelpad=12)

    plt.tight_layout()
    plt.show()


def main() -> None:
    params = MCParams(
        ticker="SPY",
        S0=500.0,
        strike_price=500.0,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        option_contracts=-1,
        sell_vol=0.32,
        hedging_vol=0.32,
        path_vol=0.32,
        greeks_vol_lookback=60,
        T_years=0.5,
        steps_per_year=252,
        n_paths=2000,
        seed=42,
        use_greek_vol_for_bands=False,
    )

    # Base scenario run
    sim = MCDeltaHedgeSimulator(params)
    results = sim.run()
    base_stats = results["final_portfolio_value"].describe(percentiles=[0.05, 0.5, 0.95])
    print("Base scenario (sell_vol vs realized path_vol) summary:")
    print(base_stats.to_string())
    # Show histogram for base scenario
    sim.plot_distribution(results)
    # Dual histogram (with vs ex-fin) if available
    if "final_portfolio_value_ex_fin" in results.columns:
        sim.plot_distribution_dual(results)

    # Sweep sell_vol to quantify vol risk premium (holding realized path_vol fixed)
    sweep_list = [0.20, 0.24, 0.28, 0.32, 0.36, 0.40]
    sweep_df = sweep_sell_vols(params, sweep_list)
    print("\nSell vol sweep vs fixed path_vol table (values in $):")
    print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:,.0f}" if abs(x) >= 1 else f"{x:.2f}"))
    # Heatmap visualization
    plot_sweep_heatmap(sweep_df)

    # --- Automatic export: Excel + Parquet ---
    print("\nGenerating detailed export (Excel + Parquet events)...")
    sim_details = sim.run_with_details(sample_quantiles=[0.05, 0.50, 0.95], return_events=True)
    excel_filename = f"mc_export_{params.ticker}_T{params.T_years:.2f}_paths{params.n_paths}.xlsx"
    try:
        with pd.ExcelWriter(excel_filename, engine="xlsxwriter") as writer:
            # Parameters sheet
            pd.DataFrame({
                "param": list(vars(params).keys()),
                "value": list(vars(params).values()),
            }).to_excel(writer, sheet_name="parameters", index=False)
            # Summary, finals, samples
            sim_details["summary"].to_excel(writer, sheet_name="summary", index=False)
            sim_details["finals"].to_excel(writer, sheet_name="path_finals", index=False)
            if sim_details.get("samples") is not None:
                sim_details["samples"].to_excel(writer, sheet_name="samples", index=False)
        print(f"Exported Excel: {excel_filename}")
    except Exception as exc:
        print(f"Excel export failed: {exc}. If missing, install XlsxWriter: pip install XlsxWriter")

    # Parquet events disabled by default. Uncomment to enable.
    # events_df = sim_details.get("events")
    # if events_df is not None and not events_df.empty:
    #     parquet_filename = f"mc_events_{params.ticker}_T{params.T_years:.2f}_paths{params.n_paths}.parquet"
    #     try:
    #         events_df.to_parquet(parquet_filename, engine="pyarrow", index=False)
    #         print(f"Exported Parquet events: {parquet_filename}")
    #     except Exception as exc:
    #         print(f"Parquet export failed: {exc}. If missing, install pyarrow: pip install pyarrow")


if __name__ == "__main__":
    main()


