from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .typing import ArrayLike
from .sizing import optimize_size


def large_hbar(labels: ArrayLike, values: ArrayLike, title: str | None = None, fig_width: float | None = None) -> Figure:
    """
    Create a large horizontal bar plot.

    Args:
        labels (ArrayLike): Labels for the bars.
        values (ArrayLike): Values for the bars.
        title (str | None, optional): Title of the plot. Defaults to None.
        fig_width (float | None, optional): Width of the figure. If None, it's calculated based on label length. Defaults to None.

    Returns:
        Figure: The matplotlib Figure object containing the horizontal bar plot.
    """
    if fig_width is None:
        max_label_len = max([len(l) for l in labels])
        fig_width = max_label_len / 8.
    
    fig, ax = plt.subplots(figsize=(fig_width, len(labels) * .2))
    
    _ = ax.barh(labels, values, zorder=5)
    ax.grid(axis='y')

    if title is not None:
        _ = ax.set_title(title)

    return optimize_size(fig, which='width')


def large_double_hbar(labels: ArrayLike, values1: pd.Series, values2: pd.Series, title: str | None = None, fig_width: float | None = None) -> Figure:
    """
    Create a large double horizontal bar plot.

    Args:
        labels (ArrayLike): Labels for the bars.
        values1 (pd.Series): First set of values for the bars.
        values2 (pd.Series): Second set of values for the bars.
        title (str | None, optional): Title of the plot. Defaults to None.
        fig_width (float | None, optional): Width of the figure. If None, it's calculated based on label length. Defaults to None.

    Returns:
        Figure: The matplotlib Figure object containing the double horizontal bar plot.
    """
    if fig_width is None:
        max_label_len = max([len(l) for l in labels])
        fig_width = max_label_len / 8.

    height = 0.5
    y = np.arange(len(labels))
    
    fig_height = len(labels) * 0.2
    y = np.linspace(0, fig_height, len(labels))
    bar_height = (y[1]-y[0]) / 2 / 1.5

    fig, ax1 = plt.subplots(figsize=(fig_width, len(labels) * .2))
    ax2 = ax1.twiny()
    
    _ = ax1.barh(y + bar_height/2, values1, bar_height, zorder=5, color='tab:blue', label=values1.name)
    _ = ax1.set_xlabel(str(values1.name), color='tab:blue')
    _ = ax1.tick_params(axis='x', labelcolor='tab:blue')

    _ = ax2.barh(y - bar_height/2, values2, bar_height, zorder=5, color='tab:orange', label=values2.name)
    _ = ax2.set_xlabel(str(values2.name), color='tab:orange')
    _ = ax2.tick_params(axis='x', labelcolor='tab:orange')

    _ = ax1.set_yticks(y)
    _ = ax1.set_yticklabels(labels)

    if title is not None:
        _ = ax1.set_title(title)

    return optimize_size(fig, which='width')
