from __future__ import annotations

import math
from abc import ABC, abstractmethod

from hedge_frontier.config import SimulationConfig, WWRebalanceMode
from hedge_frontier.portfolio import Portfolio
from hedge_frontier.pricer import Greeks


class Hedger(ABC):
    def __init__(self, config: SimulationConfig):
        self.config = config

    def reset(self) -> None:
        pass

    @abstractmethod
    def initial_hedge(self, portfolio: Portfolio, S: float, greeks: Greeks, tau: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def maybe_hedge(
        self,
        portfolio: Portfolio,
        S: float,
        greeks: Greeks,
        tau: float,
    ) -> bool:
        raise NotImplementedError


class VolatilityPriceHedger(Hedger):
    """Percentage price bands from previous day's close: S_{t-1} × (1 ± mult × σ√Δt)."""

    def __init__(self, config: SimulationConfig, spot_multiplier: float):
        super().__init__(config)
        self.spot_multiplier = float(spot_multiplier)
        self.S_prev_close: float | None = None

    def reset(self) -> None:
        self.S_prev_close = None

    @staticmethod
    def daily_move_pct(config: SimulationConfig, spot_multiplier: float) -> float:
        """One-sided daily band width as a fraction of spot (mult × σ√Δt)."""
        return spot_multiplier * config.sigma * math.sqrt(config.dt)

    def _band_bounds(self) -> tuple[float, float]:
        assert self.S_prev_close is not None
        pct = self.daily_move_pct(self.config, self.spot_multiplier)
        ref = self.S_prev_close
        return ref * (1.0 - pct), ref * (1.0 + pct)

    def initial_hedge(self, portfolio: Portfolio, S: float, greeks: Greeks, tau: float) -> None:
        target = portfolio.ideal_stock(greeks.delta)
        portfolio.execute_trade(target, S)
        self.S_prev_close = S

    def maybe_hedge(
        self,
        portfolio: Portfolio,
        S: float,
        greeks: Greeks,
        tau: float,
    ) -> bool:
        if self.S_prev_close is None:
            self.S_prev_close = S
            return False

        lower, upper = self._band_bounds()
        hedged = False
        if S < lower or S > upper:
            target = portfolio.ideal_stock(greeks.delta)
            portfolio.execute_trade(target, S)
            hedged = True

        self.S_prev_close = S
        return hedged


class DeltaValueHedger(Hedger):
    def __init__(
        self,
        config: SimulationConfig,
        risk_aversion: float,
        rebalance_mode: WWRebalanceMode | None = None,
    ):
        super().__init__(config)
        self.risk_aversion = float(risk_aversion)
        mode = rebalance_mode or config.ww_rebalance_mode
        self._hedge_to_edge = mode == WWRebalanceMode.HEDGE_TO_EDGE

    def _half_band(self, S: float, gamma: float, tau: float) -> float:
        k = self.config.transaction_cost
        gamma_risk = max(self.risk_aversion, 1e-12)
        numerator = 3.0 * math.exp(-self.config.risk_free_rate * tau) * k * S * (gamma ** 2)
        return (numerator / (2.0 * gamma_risk)) ** (1.0 / 3.0)

    def initial_hedge(self, portfolio: Portfolio, S: float, greeks: Greeks, tau: float) -> None:
        target = portfolio.ideal_stock(greeks.delta)
        portfolio.execute_trade(target, S)

    def maybe_hedge(
        self,
        portfolio: Portfolio,
        S: float,
        greeks: Greeks,
        tau: float,
    ) -> bool:
        H = self._half_band(S, greeks.gamma, tau)
        band = H * portfolio.multiplier
        ideal = portfolio.ideal_stock(greeks.delta)
        port_delta = portfolio.portfolio_delta(greeks.delta)

        upper = band
        lower = -band

        if lower <= port_delta <= upper:
            return False

        if port_delta > upper:
            target = ideal + band if self._hedge_to_edge else ideal
        else:
            target = ideal - band if self._hedge_to_edge else ideal

        portfolio.execute_trade(target, S)
        return True
