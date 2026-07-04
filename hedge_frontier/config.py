from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ConstraintMode(str, Enum):
    MAX_VARIANCE = "max_variance"
    MAX_COST = "max_cost"


class WWRebalanceMode(str, Enum):
    HEDGE_TO_EDGE = "hedge_to_edge"
    HEDGE_TO_BSM_DELTA = "hedge_to_bsm_delta"


@dataclass(frozen=True)
class SimulationConfig:
    S0: float = 100.0
    strike: float = 100.0
    sigma: float = 0.30
    risk_free_rate: float = 0.05
    T_days: int = 63
    transaction_cost: float = 0.001
    n_paths: int = 1000
    seed: int | None = 42
    steps_per_year: int = 252
    contract_multiplier: int = 100
    option_contracts: int = -1
    dividend_yield: float = 0.0
    ww_rebalance_mode: WWRebalanceMode = WWRebalanceMode.HEDGE_TO_EDGE

    @property
    def T_years(self) -> float:
        return self.T_days / self.steps_per_year

    @property
    def dt(self) -> float:
        return 1.0 / self.steps_per_year

    def cache_key(self) -> tuple:
        return (
            self.S0,
            self.strike,
            self.sigma,
            self.risk_free_rate,
            self.T_days,
            self.transaction_cost,
            self.n_paths,
            self.seed,
            self.steps_per_year,
            self.contract_multiplier,
            self.option_contracts,
            self.dividend_yield,
            self.ww_rebalance_mode.value,
        )


def simulation_config_from_cache_key(key: tuple) -> SimulationConfig:
    """Rebuild SimulationConfig from cache_key tuple (handles legacy keys)."""
    if len(key) == 12:
        return SimulationConfig(*key)
    return SimulationConfig(
        key[0],
        key[1],
        key[2],
        key[3],
        key[4],
        key[5],
        key[6],
        key[7],
        key[8],
        key[9],
        key[10],
        key[11],
        WWRebalanceMode(key[12]),
    )


@dataclass(frozen=True)
class SweepConfig:
    vol_multipliers: tuple[float, ...] = tuple(float(x) for x in np.linspace(0.1, 3.0, 15))
    gammas: tuple[float, ...] = tuple(float(x) for x in np.logspace(-2, 2, 20))


@dataclass
class SweepPoint:
    parameter: float
    mean_cost: float
    var_error: float
    mean_error: float
    n_paths: int

    @property
    def strategy_param_name(self) -> str:
        return "parameter"
