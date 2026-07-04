from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float


class OptionPricer:
    """Black-Scholes engine for a European call."""

    @staticmethod
    def _ensure_sigma(sigma: float) -> float:
        return max(1e-10, float(sigma))

    @staticmethod
    def _ensure_T(T: float) -> float:
        return max(0.0, float(T))

    @classmethod
    def greeks(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> Greeks:
        T = cls._ensure_T(T)
        if T == 0.0:
            payoff = max(S - K, 0.0)
            delta = 1.0 if S > K else 0.0
            return Greeks(price=payoff, delta=delta, gamma=0.0)

        sigma = cls._ensure_sigma(sigma)
        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = math.exp(-q * T) * norm.cdf(d1)
        gamma = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt_t)
        return Greeks(price=float(price), delta=float(delta), gamma=float(gamma))

    @classmethod
    def payoff(cls, S: float, K: float) -> float:
        return max(S - K, 0.0)

    @classmethod
    def position_premium_usd(
        cls,
        S0: float,
        strike: float,
        T_years: float,
        risk_free_rate: float,
        sigma: float,
        option_contracts: int,
        contract_multiplier: int,
        dividend_yield: float = 0.0,
    ) -> tuple[float, float]:
        """Return (position premium $, per-share BSM price) for the configured short option."""
        greeks = cls.greeks(S0, strike, T_years, risk_free_rate, sigma, dividend_yield)
        position_premium = (-option_contracts) * contract_multiplier * greeks.price
        return float(position_premium), float(greeks.price)
