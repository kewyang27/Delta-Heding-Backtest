from __future__ import annotations

from dataclasses import dataclass, field

from hedge_frontier.config import SimulationConfig
from hedge_frontier.pricer import OptionPricer


@dataclass
class Portfolio:
    config: SimulationConfig
    cash: float = 0.0
    stock: float = 0.0
    cum_tx_cost: float = 0.0
    pricer: OptionPricer = field(default_factory=OptionPricer)

    @property
    def multiplier(self) -> int:
        return self.config.contract_multiplier

    @property
    def contracts(self) -> int:
        return self.config.option_contracts

    def open_short_option(self, S: float, T: float) -> None:
        greeks = self.pricer.greeks(
            S,
            self.config.strike,
            T,
            self.config.risk_free_rate,
            self.config.sigma,
            self.config.dividend_yield,
        )
        premium = (-self.contracts) * self.multiplier * greeks.price
        self.cash += premium

    def execute_trade(self, target_stock: float, S: float) -> float:
        delta_shares = target_stock - self.stock
        if delta_shares == 0.0:
            return 0.0
        cost = abs(delta_shares) * S * self.config.transaction_cost
        self.cash -= delta_shares * S
        self.stock = target_stock
        self.cum_tx_cost += cost
        return delta_shares

    def portfolio_delta(self, bs_delta: float) -> float:
        return self.stock + self.contracts * self.multiplier * bs_delta

    def ideal_stock(self, bs_delta: float) -> float:
        return -self.contracts * self.multiplier * bs_delta

    def finalize(self, S_final: float) -> tuple[float, float]:
        self.cash += self.stock * S_final
        self.stock = 0.0
        payoff_per_share = self.pricer.payoff(S_final, self.config.strike)
        payoff_position = (-self.contracts) * self.multiplier * payoff_per_share
        hedge_value = self.cash
        hedging_error = hedge_value - payoff_position
        self.cash += self.contracts * self.multiplier * payoff_per_share
        return hedging_error, self.cum_tx_cost
