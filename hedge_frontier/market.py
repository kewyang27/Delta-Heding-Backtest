from __future__ import annotations

import math

import numpy as np

from hedge_frontier.config import SimulationConfig


class MarketEnvironment:
    """Vectorized GBM path simulator."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)

    @property
    def n_steps(self) -> int:
        return max(1, int(round(self.config.T_years * self.config.steps_per_year)))

    def simulate_paths(self) -> np.ndarray:
        cfg = self.config
        steps = self.n_steps
        dt = cfg.dt
        mu = cfg.risk_free_rate - cfg.dividend_yield
        sigma = cfg.sigma

        paths = np.empty((cfg.n_paths, steps + 1), dtype=float)
        paths[:, 0] = cfg.S0
        sqrt_dt = math.sqrt(dt)
        drift = (mu - 0.5 * sigma * sigma) * dt
        diffusion = sigma * sqrt_dt

        for t in range(1, steps + 1):
            z = np.random.normal(size=cfg.n_paths)
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * z)
        return paths
