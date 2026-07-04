from __future__ import annotations

import plotly.graph_objects as go

from hedge_frontier.config import ConstraintMode, SweepPoint, WWRebalanceMode
from hedge_frontier.metrics import replication_benchmark, tracking_std_dev_usd
from hedge_frontier.optimizer import FrontierResult, OptimalPoint


def _ww_series_suffix(mode: WWRebalanceMode) -> str:
    if mode == WWRebalanceMode.HEDGE_TO_EDGE:
        return "hedge to edge"
    return "hedge to BSM δ"


def _ww_series_name(mode: WWRebalanceMode) -> str:
    return f"Delta Value Band (WW, {_ww_series_suffix(mode)})"


def _format_param_vol(value: float) -> str:
    return f"{value:.3f}"


def _format_param_gamma(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _std_dev_series(var_series) -> list[float]:
    return [tracking_std_dev_usd(v) for v in var_series]


def _cost_vs_std_dev_hover(extra_line: str = "") -> str:
    prefix = extra_line + "<br>" if extra_line else ""
    return (
        prefix
        + "Mean Cost=$%{x:,.2f}<br>"
        + "Tracking Error Std Dev=$%{y:,.2f}<extra></extra>"
    )


def _mean_vs_std_dev_hover(extra_line: str = "") -> str:
    prefix = extra_line + "<br>" if extra_line else ""
    return (
        prefix
        + "Tracking Error Std Dev=$%{x:,.2f}<br>"
        + "Mean Tracking Error=$%{y:,.2f}<extra></extra>"
    )


def _add_bsm_benchmark_line(fig: go.Figure, bsm_premium_usd: float) -> None:
    fig.add_hline(
        y=bsm_premium_usd,
        line_dash="dot",
        line_color="#9467bd",
        annotation_text=(
            f"BSM premium sold = ${bsm_premium_usd:,.0f} "
            "(benchmark — compare std dev scale to option value)"
        ),
        annotation_position="bottom right",
    )


def _lookup_sweep_row(df, param_col: str, parameter: float):
    idx = (df[param_col] - parameter).abs().idxmin()
    return df.loc[idx]


def build_efficient_frontier_figure(
    result: FrontierResult,
    mode: ConstraintMode,
    limit: float,
    limit_std_dev_usd: float | None = None,
    ww_rebalance_mode: WWRebalanceMode = WWRebalanceMode.HEDGE_TO_EDGE,
    bsm_premium_usd: float | None = None,
    limit_rmse_usd: float | None = None,
) -> go.Figure:
    vol_df = result.vol_sweep
    gamma_df = result.gamma_sweep
    std_dev_limit = (
        limit_std_dev_usd
        if limit_std_dev_usd is not None
        else limit_rmse_usd
        if limit_rmse_usd is not None
        else tracking_std_dev_usd(limit)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=vol_df["mean_cost"],
            y=_std_dev_series(vol_df["var_error"]),
            mode="lines+markers",
            name="Volatility Price Band",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=7),
            customdata=vol_df["spot_multiplier"],
            hovertemplate=_cost_vs_std_dev_hover("Vol Multiplier=%{customdata:.3f}"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=gamma_df["mean_cost"],
            y=_std_dev_series(gamma_df["var_error"]),
            mode="lines+markers",
            name=_ww_series_name(ww_rebalance_mode),
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=7),
            customdata=gamma_df["gamma"],
            hovertemplate=_cost_vs_std_dev_hover("Gamma=%{customdata:.4f}"),
        )
    )

    _add_cost_optimal_marker(
        fig, result.optimal_vol, "Optimal Vol Band", "#1f77b4", _format_param_vol
    )
    _add_cost_optimal_marker(
        fig, result.optimal_gamma, "Optimal Delta Band", "#ff7f0e", _format_param_gamma
    )

    if mode == ConstraintMode.MAX_VARIANCE:
        fig.add_hline(
            y=std_dev_limit,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Max Tracking Error Std Dev = ${std_dev_limit:,.0f}",
            annotation_position="top right",
        )
    else:
        fig.add_vline(
            x=limit,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Max Cost = ${limit:,.0f}",
            annotation_position="top right",
        )

    if bsm_premium_usd is not None and bsm_premium_usd > 0:
        _add_bsm_benchmark_line(fig, bsm_premium_usd)

    fig.update_layout(
        title="Efficient Frontier: Replication Accuracy vs Transaction Costs",
        xaxis_title="Mean Cumulative Transaction Costs ($)",
        yaxis_title="Tracking Error Std Dev ($)",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        hovermode="closest",
        height=560,
    )
    return fig


def _axis_range(
    values,
    *,
    include_zero: bool = False,
    pad_fraction: float = 0.1,
    min_pad: float = 1.0,
    headroom_fraction: float = 0.0,
) -> tuple[float, float]:
    lo = float(min(values))
    hi = float(max(values))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    span = hi - lo
    pad = max(span * pad_fraction, min_pad)
    lo -= pad
    hi += pad + span * headroom_fraction
    return lo, hi


def build_mean_vs_std_dev_frontier_figure(
    result: FrontierResult,
    ww_rebalance_mode: WWRebalanceMode = WWRebalanceMode.HEDGE_TO_EDGE,
    limit_std_dev_usd: float | None = None,
) -> go.Figure:
    """Mean tracking error (avg P&L vs BSM) vs path-to-path std dev for each strategy."""
    vol_df = result.vol_sweep
    gamma_df = result.gamma_sweep

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=_std_dev_series(vol_df["var_error"]),
            y=vol_df["mean_error"],
            mode="lines+markers",
            name="Volatility Price Band",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=7),
            customdata=vol_df["spot_multiplier"],
            hovertemplate=_mean_vs_std_dev_hover("Vol Multiplier=%{customdata:.3f}"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=_std_dev_series(gamma_df["var_error"]),
            y=gamma_df["mean_error"],
            mode="lines+markers",
            name=_ww_series_name(ww_rebalance_mode),
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=7),
            customdata=gamma_df["gamma"],
            hovertemplate=_mean_vs_std_dev_hover("Gamma=%{customdata:.4f}"),
        )
    )

    _add_mean_std_optimal_marker(
        fig,
        result.vol_sweep,
        "spot_multiplier",
        result.optimal_vol,
        "Optimal Vol Band",
        "#1f77b4",
        _format_param_vol,
    )
    _add_mean_std_optimal_marker(
        fig,
        result.gamma_sweep,
        "gamma",
        result.optimal_gamma,
        "Optimal Delta Band",
        "#ff7f0e",
        _format_param_gamma,
    )

    _add_baseline_marker(
        fig,
        result.baseline_vol.mean_error,
        result.baseline_vol.var_error,
        "Vol mult = 1",
        "#1f77b4",
    )
    _add_baseline_marker(
        fig,
        result.baseline_gamma.mean_error,
        result.baseline_gamma.var_error,
        "γ = 1",
        "#ff7f0e",
    )

    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Break even vs BSM (mean tracking = 0)",
        annotation_position="bottom right",
    )

    if limit_std_dev_usd is not None:
        fig.add_vline(
            x=limit_std_dev_usd,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Max std dev = ${limit_std_dev_usd:,.0f}",
            annotation_position="top right",
        )

    x_values = list(_std_dev_series(vol_df["var_error"])) + list(_std_dev_series(gamma_df["var_error"]))
    y_values = list(vol_df["mean_error"]) + list(gamma_df["mean_error"])
    if limit_std_dev_usd is not None:
        x_values.append(limit_std_dev_usd)

    x_lo, x_hi = _axis_range(x_values, pad_fraction=0.06, min_pad=5.0)
    y_lo, y_hi = _axis_range(
        y_values,
        include_zero=True,
        pad_fraction=0.08,
        min_pad=1.0,
        headroom_fraction=0.22,
    )

    fig.update_layout(
        title="Efficient Frontier: Mean Tracking Error vs Replication Consistency",
        xaxis_title="Tracking Error Std Dev ($)",
        yaxis_title="Mean Tracking Error ($) — positive = favorable vs BSM",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        hovermode="closest",
        height=560,
    )
    fig.update_xaxes(range=[x_lo, x_hi])
    fig.update_yaxes(range=[y_lo, y_hi])
    return fig


def build_unit_multiplier_comparison_figure(
    baseline_vol: SweepPoint,
    baseline_gamma: SweepPoint,
    ww_rebalance_mode: WWRebalanceMode = WWRebalanceMode.HEDGE_TO_EDGE,
    bsm_premium_usd: float | None = None,
) -> go.Figure:
    """Compare both strategies at vol multiplier = 1 and risk aversion γ = 1."""
    fig = go.Figure()

    vol_std = tracking_std_dev_usd(baseline_vol.var_error)
    gamma_std = tracking_std_dev_usd(baseline_gamma.var_error)

    fig.add_trace(
        go.Scatter(
            x=[baseline_vol.mean_cost],
            y=[vol_std],
            mode="markers+text",
            name="Volatility Price Band (mult = 1)",
            marker=dict(size=20, symbol="circle", color="#1f77b4", line=dict(width=2, color="black")),
            text=["Vol mult = 1"],
            textposition="top center",
            hovertemplate=_cost_vs_std_dev_hover("Vol Multiplier=1.000"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[baseline_gamma.mean_cost],
            y=[gamma_std],
            mode="markers+text",
            name=f"Delta Value Band (γ = 1, {_ww_series_suffix(ww_rebalance_mode)})",
            marker=dict(size=20, symbol="circle", color="#ff7f0e", line=dict(width=2, color="black")),
            text=["γ = 1"],
            textposition="top center",
            hovertemplate=_cost_vs_std_dev_hover("Gamma=1.0000"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[baseline_vol.mean_cost, baseline_gamma.mean_cost],
            y=[vol_std, gamma_std],
            mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    if bsm_premium_usd is not None and bsm_premium_usd > 0:
        benchmark = replication_benchmark(bsm_premium_usd, baseline_vol, baseline_gamma)
        fig.add_hline(
            y=benchmark.vol_band.tracking_std_dev,
            line_dash="dot",
            line_color="#1f77b4",
            annotation_text=(
                f"Vol band std dev = ${benchmark.vol_band.tracking_std_dev:,.0f}"
            ),
            annotation_position="bottom left",
        )
        fig.add_hline(
            y=benchmark.delta_band.tracking_std_dev,
            line_dash="dot",
            line_color="#ff7f0e",
            annotation_text=(
                f"Delta band std dev = ${benchmark.delta_band.tracking_std_dev:,.0f}"
            ),
            annotation_position="top left",
        )

    fig.update_layout(
        title="Baseline Comparison at Unit Multiplier (Vol Mult = 1, γ = 1)",
        xaxis_title="Mean Cumulative Transaction Costs ($)",
        yaxis_title="Tracking Error Std Dev ($)",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        hovermode="closest",
        height=480,
    )
    return fig


def _add_cost_optimal_marker(
    fig: go.Figure,
    point: OptimalPoint,
    name: str,
    color: str,
    formatter,
) -> None:
    if not point.feasible:
        return
    fig.add_trace(
        go.Scatter(
            x=[point.mean_cost],
            y=[tracking_std_dev_usd(point.var_error)],
            mode="markers",
            name=name,
            marker=dict(size=18, symbol="diamond", color=color, line=dict(width=2, color="black")),
            hovertemplate=(
                f"Parameter={formatter(point.parameter)}<br>"
                "Mean Cost=$%{x:,.2f}<br>"
                "Tracking Error Std Dev=$%{y:,.2f}<extra></extra>"
            ),
        )
    )


def _add_mean_std_optimal_marker(
    fig: go.Figure,
    sweep_df,
    param_col: str,
    point: OptimalPoint,
    name: str,
    color: str,
    formatter,
) -> None:
    if not point.feasible:
        return
    row = _lookup_sweep_row(sweep_df, param_col, point.parameter)
    fig.add_trace(
        go.Scatter(
            x=[tracking_std_dev_usd(row["var_error"])],
            y=[row["mean_error"]],
            mode="markers",
            name=name,
            marker=dict(size=18, symbol="diamond", color=color, line=dict(width=2, color="black")),
            hovertemplate=(
                f"Parameter={formatter(point.parameter)}<br>"
                "Tracking Error Std Dev=$%{x:,.2f}<br>"
                "Mean Tracking Error=$%{y:,.2f}<extra></extra>"
            ),
        )
    )


def _add_baseline_marker(
    fig: go.Figure,
    mean_error: float,
    var_error: float,
    label: str,
    color: str,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=[tracking_std_dev_usd(var_error)],
            y=[mean_error],
            mode="markers+text",
            name=f"Baseline ({label})",
            marker=dict(size=14, symbol="circle", color=color, line=dict(width=2, color="black")),
            text=[label],
            textposition="top center",
            hovertemplate=_mean_vs_std_dev_hover(f"Baseline {label}"),
        )
    )
