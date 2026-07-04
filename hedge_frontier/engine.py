from __future__ import annotations

import numpy as np

from hedge_frontier.config import SimulationConfig, SweepPoint
from hedge_frontier.hedgers import Hedger
from hedge_frontier.market import MarketEnvironment
from hedge_frontier.portfolio import Portfolio
from hedge_frontier.pricer import OptionPricer


class SimulationEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.market = MarketEnvironment(config)
        self.pricer = OptionPricer()

    def run(self, hedger: Hedger, paths: np.ndarray | None = None) -> SweepPoint:
        if paths is None:
            paths = self.market.simulate_paths()

        cfg = self.config
        steps = paths.shape[1] - 1
        errors = np.empty(cfg.n_paths, dtype=float)
        costs = np.empty(cfg.n_paths, dtype=float)

        for i in range(cfg.n_paths):
            hedger.reset()
            portfolio = Portfolio(cfg)
            price_path = paths[i]

            S0 = float(price_path[0])
            tau0 = cfg.T_years
            greeks0 = self.pricer.greeks(
                S0,
                cfg.strike,
                tau0,
                cfg.risk_free_rate,
                cfg.sigma,
                cfg.dividend_yield,
            )
            portfolio.open_short_option(S0, tau0)
            hedger.initial_hedge(portfolio, S0, greeks0, tau0)

            for t in range(1, steps + 1):
                S = float(price_path[t])
                tau = max(0.0, cfg.T_years - t * cfg.dt)
                greeks = self.pricer.greeks(
                    S,
                    cfg.strike,
                    tau,
                    cfg.risk_free_rate,
                    cfg.sigma,
                    cfg.dividend_yield,
                )
                hedger.maybe_hedge(portfolio, S, greeks, tau)

            S_final = float(price_path[-1])
            error, cost = portfolio.finalize(S_final)
            errors[i] = error
            costs[i] = cost

        return SweepPoint(
            parameter=getattr(hedger, "spot_multiplier", getattr(hedger, "risk_aversion", 0.0)),
            mean_cost=float(np.mean(costs)),
            var_error=float(np.var(errors, ddof=1)) if cfg.n_paths > 1 else 0.0,
            mean_error=float(np.mean(errors)),
            n_paths=cfg.n_paths,
        )
