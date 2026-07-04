from __future__ import annotations

import math
from dataclasses import dataclass

from hedge_frontier.config import SweepPoint


def tracking_std_dev_usd(var_error: float) -> float:
    """Standard deviation of per-path tracking errors (√variance)."""
    return math.sqrt(max(0.0, float(var_error)))


def rmse_usd(var_error: float) -> float:
    """Backward-compatible alias for tracking_std_dev_usd."""
    return tracking_std_dev_usd(var_error)


def variance_limit_from_std_dev(std_dev_usd: float) -> float:
    """Convert user-facing tracking-error std dev ($) limit to variance ($²)."""
    value = max(0.0, float(std_dev_usd))
    return value * value


def variance_limit_from_rmse(rmse_usd: float) -> float:
    """Backward-compatible alias for variance_limit_from_std_dev."""
    return variance_limit_from_std_dev(rmse_usd)


def replication_value_usd(bsm_premium_usd: float, mean_tracking_error: float) -> float:
    """Effective hedge replication at expiry vs the BSM premium sold."""
    return bsm_premium_usd + mean_tracking_error


def baseline_slippage_std_dev_usd(baseline_vol: SweepPoint, baseline_gamma: SweepPoint) -> float:
    """Average tracking-error std dev at vol mult = 1 and WW γ = 1."""
    return 0.5 * (
        tracking_std_dev_usd(baseline_vol.var_error)
        + tracking_std_dev_usd(baseline_gamma.var_error)
    )


def baseline_slippage_rmse_usd(baseline_vol: SweepPoint, baseline_gamma: SweepPoint) -> float:
    return baseline_slippage_std_dev_usd(baseline_vol, baseline_gamma)


@dataclass(frozen=True)
class StrategyReplication:
    mean_tracking_error: float
    tracking_std_dev: float

    @property
    def tracking_rmse(self) -> float:
        return self.tracking_std_dev

    @property
    def tracking_label(self) -> str:
        if self.mean_tracking_error >= 0:
            return f"+${self.mean_tracking_error:,.2f} (favorable)"
        return f"-${abs(self.mean_tracking_error):,.2f} (against you)"

    def replication_value_usd(self, bsm_premium_usd: float) -> float:
        return replication_value_usd(bsm_premium_usd, self.mean_tracking_error)


@dataclass(frozen=True)
class ReplicationBenchmark:
    """Compare discrete-hedge replication outcomes to the BSM premium sold."""

    bsm_premium_usd: float
    vol_band: StrategyReplication
    delta_band: StrategyReplication


def replication_benchmark(
    bsm_premium_usd: float,
    baseline_vol: SweepPoint,
    baseline_gamma: SweepPoint,
) -> ReplicationBenchmark:
    return ReplicationBenchmark(
        bsm_premium_usd=bsm_premium_usd,
        vol_band=StrategyReplication(
            mean_tracking_error=baseline_vol.mean_error,
            tracking_std_dev=tracking_std_dev_usd(baseline_vol.var_error),
        ),
        delta_band=StrategyReplication(
            mean_tracking_error=baseline_gamma.mean_error,
            tracking_std_dev=tracking_std_dev_usd(baseline_gamma.var_error),
        ),
    )
