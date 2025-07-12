from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go

from .analysis import derivative_curves

__all__ = ["plot_capacity", "plot_dqdv"]


def plot_capacity(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["cycle_index"], y=df["capacity_mAh"], mode="lines"))
    fig.update_layout(title="Capacity vs Cycle", xaxis_title="Cycle", yaxis_title="Capacity (mAh)")
    return fig


def plot_dqdv(df: pd.DataFrame, cycles: list[int] | None = None) -> go.Figure:
    if cycles is None:
        cycles = [1, 50, 100]
    curves = derivative_curves(df, cycles=cycles, kind="dQdV")
    fig = go.Figure()
    for cyc, cdf in curves.items():
        fig.add_trace(go.Scatter(x=cdf["V"], y=cdf["dQ/dV"], mode="lines", name=f"Cycle {cyc}"))
    fig.update_layout(title="dQ/dV Curves", xaxis_title="Voltage (V)", yaxis_title="dQ/dV (mAh/V)")
    return fig 