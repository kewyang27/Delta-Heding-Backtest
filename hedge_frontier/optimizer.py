from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hedge_frontier.config import ConstraintMode, SimulationConfig, SweepConfig, SweepPoint
from hedge_frontier.engine import SimulationEngine
from hedge_frontier.fast_engine import precompute_paths, run_gamma_sweep_point, run_vol_sweep_point


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class OptimalPoint:
    parameter: float
    mean_cost: float
    var_error: float
    feasible: bool


@dataclass
class FrontierResult:
    vol_sweep: pd.DataFrame
    gamma_sweep: pd.DataFrame
    optimal_vol: OptimalPoint
    optimal_gamma: OptimalPoint
    baseline_vol: SweepPoint
    baseline_gamma: SweepPoint


class FrontierOptimizer:
    def __init__(self, config: SimulationConfig, sweep_config: SweepConfig | None = None):
        self.config = config
        self.sweep_config = sweep_config or SweepConfig()
        self.engine = SimulationEngine(config)
        self._paths = self.engine.market.simulate_paths()
        self._precomputed = precompute_paths(config, self._paths)

    def sweep_vol_multipliers(self, progress: ProgressCallback | None = None) -> pd.DataFrame:
        rows: list[dict] = []
        multipliers = self.sweep_config.vol_multipliers
        total = len(multipliers)
        for idx, mult in enumerate(multipliers, start=1):
            point = run_vol_sweep_point(self.config, self._precomputed, mult)
            rows.append(
                {
                    "spot_multiplier": mult,
                    "mean_cost": point.mean_cost,
                    "var_error": point.var_error,
                    "mean_error": point.mean_error,
                }
            )
            if progress is not None:
                progress(idx, total, f"Volatility band sweep ({idx}/{total})")
        return pd.DataFrame(rows).sort_values("spot_multiplier").reset_index(drop=True)

    def sweep_gammas(self, progress: ProgressCallback | None = None) -> pd.DataFrame:
        rows: list[dict] = []
        gammas = self.sweep_config.gammas
        total = len(gammas)
        for idx, gamma in enumerate(gammas, start=1):
            point = run_gamma_sweep_point(self.config, self._precomputed, gamma)
            rows.append(
                {
                    "gamma": gamma,
                    "mean_cost": point.mean_cost,
                    "var_error": point.var_error,
                    "mean_error": point.mean_error,
                }
            )
            if progress is not None:
                progress(idx, total, f"Delta band sweep ({idx}/{total})")
        return pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)

    def run_frontier(self) -> FrontierResult:
        vol_df = self.sweep_vol_multipliers()
        gamma_df = self.sweep_gammas()
        return FrontierResult(
            vol_sweep=vol_df,
            gamma_sweep=gamma_df,
            optimal_vol=OptimalPoint(0.0, 0.0, 0.0, False),
            optimal_gamma=OptimalPoint(0.0, 0.0, 0.0, False),
            baseline_vol=run_vol_sweep_point(self.config, self._precomputed, 1.0),
            baseline_gamma=run_gamma_sweep_point(self.config, self._precomputed, 1.0),
        )


def select_optimal(
    df: pd.DataFrame,
    param_col: str,
    mode: ConstraintMode,
    limit: float,
) -> OptimalPoint:
    if df.empty:
        return OptimalPoint(0.0, 0.0, 0.0, False)

    if mode == ConstraintMode.MAX_VARIANCE:
        feasible = df[df["var_error"] < limit]
        if feasible.empty:
            return OptimalPoint(0.0, 0.0, 0.0, False)
        best = feasible.loc[feasible["mean_cost"].idxmin()]
    else:
        feasible = df[df["mean_cost"] < limit]
        if feasible.empty:
            return OptimalPoint(0.0, 0.0, 0.0, False)
        best = feasible.loc[feasible["var_error"].idxmin()]

    return OptimalPoint(
        parameter=float(best[param_col]),
        mean_cost=float(best["mean_cost"]),
        var_error=float(best["var_error"]),
        feasible=True,
    )


def compute_baseline_points(
    config: SimulationConfig,
    sweep_config: SweepConfig | None = None,
) -> tuple[SweepPoint, SweepPoint]:
    optimizer = FrontierOptimizer(config, sweep_config)
    vol_point = run_vol_sweep_point(config, optimizer._precomputed, 1.0)
    gamma_point = run_gamma_sweep_point(config, optimizer._precomputed, 1.0)
    return vol_point, gamma_point


def run_full_analysis(
    config: SimulationConfig,
    sweep_config: SweepConfig,
    mode: ConstraintMode,
    limit: float,
    progress: ProgressCallback | None = None,
) -> FrontierResult:
    optimizer = FrontierOptimizer(config, sweep_config)
    vol_total = len(sweep_config.vol_multipliers)
    gamma_total = len(sweep_config.gammas)
    combined_total = vol_total + gamma_total

    def _progress(local_idx: int, local_total: int, label: str) -> None:
        if progress is None:
            return
        if "Volatility" in label:
            global_idx = local_idx
        else:
            global_idx = vol_total + local_idx
        progress(global_idx, combined_total, label)

    vol_df = optimizer.sweep_vol_multipliers(progress=_progress)
    gamma_df = optimizer.sweep_gammas(progress=_progress)
    optimal_vol = select_optimal(vol_df, "spot_multiplier", mode, limit)
    optimal_gamma = select_optimal(gamma_df, "gamma", mode, limit)
    baseline_vol = run_vol_sweep_point(config, optimizer._precomputed, 1.0)
    baseline_gamma = run_gamma_sweep_point(config, optimizer._precomputed, 1.0)
    return FrontierResult(
        vol_sweep=vol_df,
        gamma_sweep=gamma_df,
        optimal_vol=optimal_vol,
        optimal_gamma=optimal_gamma,
        baseline_vol=baseline_vol,
        baseline_gamma=baseline_gamma,
    )
