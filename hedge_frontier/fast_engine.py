from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from hedge_frontier.config import SimulationConfig, SweepPoint, WWRebalanceMode


@dataclass
class PrecomputedPaths:
    paths: np.ndarray
    deltas: np.ndarray
    gammas: np.ndarray
    taus: np.ndarray
    prices: np.ndarray


def _call_greeks_vectorized(
    S: np.ndarray,
    K: float,
    tau: np.ndarray,
    r: float,
    sigma: float,
    q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = np.empty_like(S, dtype=float)
    deltas = np.empty_like(S, dtype=float)
    gammas = np.empty_like(S, dtype=float)

    expired = tau <= 0.0
    if np.any(expired):
        prices[expired] = np.maximum(S[expired] - K, 0.0)
        deltas[expired] = np.where(S[expired] > K, 1.0, 0.0)
        gammas[expired] = 0.0

    live = ~expired
    if np.any(live):
        s_live = S[live]
        t_live = tau[live]
        sig = max(1e-10, sigma)
        sqrt_t = np.sqrt(t_live)
        d1 = (np.log(s_live / K) + (r - q + 0.5 * sig * sig) * t_live) / (sig * sqrt_t)
        d2 = d1 - sig * sqrt_t
        prices[live] = (
            s_live * np.exp(-q * t_live) * norm.cdf(d1)
            - K * np.exp(-r * t_live) * norm.cdf(d2)
        )
        deltas[live] = np.exp(-q * t_live) * norm.cdf(d1)
        gammas[live] = np.exp(-q * t_live) * norm.pdf(d1) / (s_live * sig * sqrt_t)

    return prices, deltas, gammas


def precompute_paths(config: SimulationConfig, paths: np.ndarray) -> PrecomputedPaths:
    n_paths, n_steps = paths.shape
    dt = config.dt
    time_idx = np.arange(n_steps, dtype=float)
    tau_row = np.maximum(0.0, config.T_years - time_idx * dt)
    taus = np.broadcast_to(tau_row, (n_paths, n_steps))

    deltas = np.empty((n_paths, n_steps), dtype=float)
    gammas = np.empty((n_paths, n_steps), dtype=float)
    prices = np.empty((n_paths, n_steps), dtype=float)

    for i in range(n_paths):
        p, d, g = _call_greeks_vectorized(
            paths[i],
            config.strike,
            tau_row,
            config.risk_free_rate,
            config.sigma,
            config.dividend_yield,
        )
        prices[i] = p
        deltas[i] = d
        gammas[i] = g

    return PrecomputedPaths(paths=paths, deltas=deltas, gammas=gammas, taus=taus, prices=prices)


def _apply_trade(
    stock: float,
    cash: float,
    cum_cost: float,
    target: float,
    S: float,
    k: float,
) -> tuple[float, float, float]:
    delta_shares = target - stock
    if delta_shares == 0.0:
        return stock, cash, cum_cost
    cost = abs(delta_shares) * S * k
    cash -= delta_shares * S
    return target, cash, cum_cost + cost


def _finalize_path(
    stock: float,
    cash: float,
    S_final: float,
    strike: float,
    contracts: int,
    multiplier: int,
) -> tuple[float, float]:
    cash += stock * S_final
    payoff_per_share = max(S_final - strike, 0.0)
    payoff_position = (-contracts) * multiplier * payoff_per_share
    hedging_error = cash - payoff_position
    return hedging_error, cash


def run_vol_sweep_point(
    config: SimulationConfig,
    data: PrecomputedPaths,
    spot_multiplier: float,
) -> SweepPoint:
    cfg = config
    k = cfg.transaction_cost
    mult = cfg.contract_multiplier
    contracts = cfg.option_contracts
    dt = cfg.dt
    sigma = cfg.sigma
    sqrt_dt = math.sqrt(dt)
    n_paths, n_steps = data.paths.shape

    errors = np.empty(n_paths, dtype=float)
    costs = np.empty(n_paths, dtype=float)

    for i in range(n_paths):
        S = data.paths[i]
        deltas = data.deltas[i]
        S0 = S[0]

        cash = (-contracts) * mult * data.prices[i, 0]
        target0 = -contracts * mult * deltas[0]
        stock, cash, cum_cost = _apply_trade(0.0, cash, 0.0, target0, S0, k)

        daily_pct = spot_multiplier * sigma * sqrt_dt
        for t in range(1, n_steps):
            prev_close = S[t - 1]
            St = S[t]
            lower = prev_close * (1.0 - daily_pct)
            upper = prev_close * (1.0 + daily_pct)
            if St < lower or St > upper:
                target = -contracts * mult * deltas[t]
                stock, cash, cum_cost = _apply_trade(stock, cash, cum_cost, target, St, k)

        errors[i], _ = _finalize_path(stock, cash, S[-1], cfg.strike, contracts, mult)
        costs[i] = cum_cost

    return SweepPoint(
        parameter=spot_multiplier,
        mean_cost=float(np.mean(costs)),
        var_error=float(np.var(errors, ddof=1)) if n_paths > 1 else 0.0,
        mean_error=float(np.mean(errors)),
        n_paths=n_paths,
    )


def run_gamma_sweep_point(
    config: SimulationConfig,
    data: PrecomputedPaths,
    risk_aversion: float,
) -> SweepPoint:
    cfg = config
    k = cfg.transaction_cost
    mult = cfg.contract_multiplier
    contracts = cfg.option_contracts
    gamma_risk = max(risk_aversion, 1e-12)
    r = cfg.risk_free_rate
    n_paths, n_steps = data.paths.shape

    errors = np.empty(n_paths, dtype=float)
    costs = np.empty(n_paths, dtype=float)

    for i in range(n_paths):
        S = data.paths[i]
        deltas = data.deltas[i]
        gammas = data.gammas[i]
        taus = data.taus[i]
        S0 = S[0]

        cash = (-contracts) * mult * data.prices[i, 0]
        target0 = -contracts * mult * deltas[0]
        stock, cash, cum_cost = _apply_trade(0.0, cash, 0.0, target0, S0, k)

        for t in range(1, n_steps):
            St = S[t]
            gamma_opt = gammas[t]
            tau = taus[t]
            numerator = 3.0 * math.exp(-r * tau) * k * St * (gamma_opt ** 2)
            band = (numerator / (2.0 * gamma_risk)) ** (1.0 / 3.0) * mult

            ideal = -contracts * mult * deltas[t]
            port_delta = stock + contracts * mult * deltas[t]

            if -band <= port_delta <= band:
                continue

            hedge_to_edge = cfg.ww_rebalance_mode == WWRebalanceMode.HEDGE_TO_EDGE
            if port_delta > band:
                target = ideal + band if hedge_to_edge else ideal
            else:
                target = ideal - band if hedge_to_edge else ideal
            stock, cash, cum_cost = _apply_trade(stock, cash, cum_cost, target, St, k)

        errors[i], _ = _finalize_path(stock, cash, S[-1], cfg.strike, contracts, mult)
        costs[i] = cum_cost

    return SweepPoint(
        parameter=risk_aversion,
        mean_cost=float(np.mean(costs)),
        var_error=float(np.var(errors, ddof=1)) if n_paths > 1 else 0.0,
        mean_error=float(np.mean(errors)),
        n_paths=n_paths,
    )
