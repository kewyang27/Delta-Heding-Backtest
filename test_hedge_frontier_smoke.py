"""Manual smoke tests for hedge_frontier package."""

from __future__ import annotations

import numpy as np

from hedge_frontier.config import ConstraintMode, SimulationConfig, SweepConfig, WWRebalanceMode
from hedge_frontier.engine import SimulationEngine
from hedge_frontier.hedgers import DeltaValueHedger, VolatilityPriceHedger
from hedge_frontier.optimizer import run_full_analysis, select_optimal


def test_vol_monotonicity() -> None:
    config = SimulationConfig(
        S0=100.0,
        strike=100.0,
        sigma=0.30,
        risk_free_rate=0.05,
        T_days=63,
        transaction_cost=0.001,
        n_paths=500,
        seed=42,
    )
    engine = SimulationEngine(config)
    paths = engine.market.simulate_paths()
    multipliers = [0.2, 0.5, 1.0, 2.0, 3.0]
    costs = []
    variances = []
    for mult in multipliers:
        point = engine.run(VolatilityPriceHedger(config, mult), paths=paths)
        costs.append(point.mean_cost)
        variances.append(point.var_error)
    assert costs[0] > costs[-1], f"Expected lower cost at high multiplier: {costs}"
    assert variances[-1] > variances[0], f"Expected higher variance at wide bands: {variances}"
    print("PASS: vol multiplier monotonicity")


def test_gamma_monotonicity() -> None:
    config = SimulationConfig(
        S0=100.0,
        strike=100.0,
        sigma=0.30,
        risk_free_rate=0.05,
        T_days=63,
        transaction_cost=0.001,
        n_paths=500,
        seed=42,
    )
    engine = SimulationEngine(config)
    paths = engine.market.simulate_paths()
    gammas = [0.01, 0.1, 1.0, 10.0, 100.0]
    costs = []
    variances = []
    for gamma in gammas:
        point = engine.run(DeltaValueHedger(config, gamma), paths=paths)
        costs.append(point.mean_cost)
        variances.append(point.var_error)
    assert costs[-1] > costs[0], f"Expected higher cost at high gamma: {costs}"
    print("PASS: gamma cost monotonicity")


def test_hedge_to_edge() -> None:
    config = SimulationConfig(
        S0=100.0,
        strike=100.0,
        sigma=0.30,
        risk_free_rate=0.05,
        T_days=63,
        transaction_cost=0.001,
        n_paths=1,
        seed=7,
        ww_rebalance_mode=WWRebalanceMode.HEDGE_TO_EDGE,
    )
    engine = SimulationEngine(config)
    paths = engine.market.simulate_paths()
    hedger = DeltaValueHedger(config, risk_aversion=1.0)

    from hedge_frontier.portfolio import Portfolio
    from hedge_frontier.pricer import OptionPricer

    pricer = OptionPricer()
    portfolio = Portfolio(config)
    price_path = paths[0]
    S0 = float(price_path[0])
    greeks0 = pricer.greeks(S0, config.strike, config.T_years, config.risk_free_rate, config.sigma)
    portfolio.open_short_option(S0, config.T_years)
    hedger.initial_hedge(portfolio, S0, greeks0, config.T_years)

    edge_hits = 0
    steps = paths.shape[1] - 1
    for t in range(1, steps + 1):
        S = float(price_path[t])
        tau = max(0.0, config.T_years - t * config.dt)
        greeks = pricer.greeks(S, config.strike, tau, config.risk_free_rate, config.sigma)
        before = portfolio.stock
        triggered = hedger.maybe_hedge(portfolio, S, greeks, tau)
        if triggered:
            H = hedger._half_band(S, greeks.gamma, tau) * config.contract_multiplier
            after_delta = portfolio.portfolio_delta(greeks.delta)
            assert abs(abs(after_delta) - H) < 1e-4, f"Expected edge hedge, got delta={after_delta}, H={H}"
            edge_hits += 1
    print(f"PASS: hedge-to-edge ({edge_hits} hedges checked)")


def test_hedge_to_bsm_delta() -> None:
    config = SimulationConfig(
        S0=100.0,
        strike=100.0,
        sigma=0.30,
        risk_free_rate=0.05,
        T_days=63,
        transaction_cost=0.001,
        n_paths=1,
        seed=7,
        ww_rebalance_mode=WWRebalanceMode.HEDGE_TO_BSM_DELTA,
    )
    engine = SimulationEngine(config)
    paths = engine.market.simulate_paths()

    from hedge_frontier.portfolio import Portfolio
    from hedge_frontier.pricer import OptionPricer

    pricer = OptionPricer()
    portfolio = Portfolio(config)
    price_path = paths[0]
    S0 = float(price_path[0])
    greeks0 = pricer.greeks(S0, config.strike, config.T_years, config.risk_free_rate, config.sigma)
    portfolio.open_short_option(S0, config.T_years)
    hedger = DeltaValueHedger(config, risk_aversion=1.0)
    hedger.initial_hedge(portfolio, S0, greeks0, config.T_years)

    delta_hits = 0
    steps = paths.shape[1] - 1
    for t in range(1, steps + 1):
        S = float(price_path[t])
        tau = max(0.0, config.T_years - t * config.dt)
        greeks = pricer.greeks(S, config.strike, tau, config.risk_free_rate, config.sigma)
        triggered = hedger.maybe_hedge(portfolio, S, greeks, tau)
        if triggered:
            after_delta = portfolio.portfolio_delta(greeks.delta)
            assert abs(after_delta) < 1e-4, f"Expected full delta hedge, got delta={after_delta}"
            delta_hits += 1
    print(f"PASS: hedge-to-BSM-delta ({delta_hits} hedges checked)")


def test_optimizer() -> None:
    config = SimulationConfig(
        S0=100.0,
        strike=100.0,
        sigma=0.30,
        risk_free_rate=0.05,
        T_days=63,
        transaction_cost=0.001,
        n_paths=300,
        seed=42,
    )
    sweep = SweepConfig(
        vol_multipliers=tuple(np.linspace(0.2, 2.5, 10)),
        gammas=tuple(np.logspace(-1, 1, 10)),
    )
    result = run_full_analysis(config, sweep, ConstraintMode.MAX_COST, limit=5000.0)
    assert result.optimal_vol.feasible or result.optimal_gamma.feasible
    if result.optimal_vol.feasible:
        row = result.vol_sweep[result.vol_sweep["spot_multiplier"] == result.optimal_vol.parameter].iloc[0]
        assert row["mean_cost"] < 5000.0
    print("PASS: optimizer max cost")

    result2 = run_full_analysis(config, sweep, ConstraintMode.MAX_VARIANCE, limit=1e8)
    assert result2.optimal_vol.feasible
    print("PASS: optimizer max variance")


if __name__ == "__main__":
    test_vol_monotonicity()
    test_gamma_monotonicity()
    test_hedge_to_edge()
    test_hedge_to_bsm_delta()
    test_optimizer()
    print("All smoke tests passed.")
