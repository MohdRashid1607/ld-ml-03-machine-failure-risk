"""
Evaluation plots for anomaly detection results.

Generates Plotly figures for time-series anomaly visualization,
score distributions, and machine-level comparisons.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


def plot_anomaly_timeseries(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    datetime_col: str = "draw_local_datetime",
    machine_col: str = "machine_id",
    title: str = "Anomaly Score Time Series",
) -> go.Figure:
    """
    Plot anomaly scores over time with anomaly markers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least a datetime column.
    scores : np.ndarray
        1D array of anomaly scores matching df's row order.
    threshold : float
        Score threshold above which anomalies are highlighted.
    datetime_col : str
        Name of the datetime column in df.
    machine_col : str
        Name of the machine ID column (used for hover info).
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    plot_df = df[[datetime_col, machine_col]].copy() if machine_col in df.columns else df[[datetime_col]].copy()
    plot_df["anomaly_score"] = scores
    plot_df["is_anomaly"] = scores > threshold

    normal = plot_df[~plot_df["is_anomaly"]]
    anomalies = plot_df[plot_df["is_anomaly"]]

    fig = go.Figure()

    # Normal points
    fig.add_trace(go.Scatter(
        x=normal[datetime_col],
        y=normal["anomaly_score"],
        mode="lines+markers",
        name="Normal",
        line=dict(color="#4C9BE8", width=1.5),
        marker=dict(size=4, color="#4C9BE8"),
        hovertemplate="%{x}<br>Score: %{y:.3f}<extra>Normal</extra>",
    ))

    # Anomaly points
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies[datetime_col],
            y=anomalies["anomaly_score"],
            mode="markers",
            name="Anomaly",
            marker=dict(size=9, color="#FF4B4B", symbol="x"),
            hovertemplate="%{x}<br>Score: %{y:.3f}<extra>⚠️ Anomaly</extra>",
        ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#FFA500",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date / Time",
        yaxis_title="Anomaly Score",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=400,
    )
    return fig


def plot_score_distribution(
    scores: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Anomaly Score Distribution",
) -> go.Figure:
    """
    Plot a histogram of anomaly scores with an optional threshold line.

    Parameters
    ----------
    scores : np.ndarray
        1D array of anomaly scores.
    threshold : float, optional
        If provided, draw a vertical threshold line.
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=50,
        name="Score Distribution",
        marker_color="#4C9BE8",
        opacity=0.75,
    ))

    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#FFA500",
            annotation_text=f"Threshold ({threshold:.2f})",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        template="plotly_dark",
        bargap=0.05,
        height=350,
    )
    return fig


def plot_machine_risk_bar(
    df: pd.DataFrame,
    machine_col: str = "machine_id",
    score_col: str = "anomaly_score",
    title: str = "Average Anomaly Score by Machine",
) -> go.Figure:
    """
    Bar chart of average anomaly score per machine for comparative analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with machine_id and anomaly_score columns.
    machine_col : str
        Column name for machine identifiers.
    score_col : str
        Column name for anomaly scores.
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    machine_scores = (
        df.groupby(machine_col)[score_col]
        .mean()
        .reset_index()
        .sort_values(score_col, ascending=False)
    )

    fig = px.bar(
        machine_scores,
        x=machine_col,
        y=score_col,
        color=score_col,
        color_continuous_scale="RdYlGn_r",
        title=title,
        template="plotly_dark",
        labels={score_col: "Mean Anomaly Score", machine_col: "Machine ID"},
    )
    fig.update_layout(height=400)
    return fig
