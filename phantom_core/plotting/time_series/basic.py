import numpy as np
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from datetime import timedelta, datetime
import warnings

from ..annotations import annotate_ax
from ...constants import DEFAULT_DATE_FORMAT_STR


def compare_two_timeseries_different_scales_plot(df: pd.DataFrame | None = None, cname1: str | None = None, cname2: str | None = None, s1: pd.Series | None = None, s2: pd.Series | None = None):

    if df is not None and cname1 is not None and cname2 is not None:
        s1 = df[cname1]
        s2 = df[cname2]
    elif s1 is not None and s2 is not None:
        pass
    else:
        raise ValueError('passed invalid combination of arguments')


    trace1 = go.Scatter(
        x=s1.index,
        y=s1,
        name=s1.name,
        yaxis='y1'
    )

    # Create the second trace for 'close'
    trace2 = go.Scatter(
        x=s2.index,
        y=s2,
        name=s2.name,
        yaxis='y2'
    )

    # Create the layout with two y-axes
    layout = go.Layout(
        title=f'Comparing {s1.name} and {s2.name}',
        xaxis=dict(title='Date'),
        yaxis=dict(
            title=cname1,
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title=cname2,
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        ),
        height=1200,
    )

    # Create the figure with data and layout
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    fig.update_layout(hovermode='x unified')

    # Show the plot
    fig.show()


class VisualizeIntradayMultipleTS:
    """
    Visualize multiple columns of an intra-day time series. Allows you to specify a start date and 
    a number of days to look ahead. This class will figure out how to slice the dataframe appropriately
    and visualize the time series without any weirdness around non-trading days.

    Also handles cases where the requested start date is not a trading day.

    Unfortunatly, does not label the x-axes with timestamps but with range index due to the non-trading
    day weirdness.

    Example usage:

    ```
    viz = VisualizeIntradayMultipleTS(ohlcv)
    cnames = ['close', 'atr64', 'tp']
    start = '2024-07-24'
    days_ahead = 1

    viz.visualize(cnames, start, days_ahead)
    ```
    """

    def __init__(self, df: pd.DataFrame):
        assert isinstance(df.index, pd.DatetimeIndex)
        self.df = df
        self.days = [d.strftime(DEFAULT_DATE_FORMAT_STR) for d in df.index.normalize().unique().tolist()]


    def _get_start_center_end_days(self, date: str | datetime | None, days_behind: int, days_ahead: int | None) -> tuple[str, str, str]:
        
        # resolve center date via types
        if date is None:
            # if no date it means they want to start at the begniing
            center_day = self.df.index[0].date().strftime(DEFAULT_DATE_FORMAT_STR)
        elif isinstance(date, datetime):
            center_day = date.strftime(DEFAULT_DATE_FORMAT_STR)
        else:
            center_day = date

        # find the first day in self.days on or after the center date
        assert center_day is not None
        center_ts = pd.to_datetime(center_day)
        while center_day not in self.days:
            center_ts += pd.Timedelta(days=1)
            center_day = center_ts.strftime(DEFAULT_DATE_FORMAT_STR)
        center_day_idx = self.days.index(center_day)

        # start day is center day backed up `days_behind` number of days
        start_day = self.days[center_day_idx - days_behind]

        if days_ahead is None:
            # means they want everything after the center date
            end_day = self.days[-1]
        else:
            end_day = self.days[center_day_idx + days_ahead]

        return start_day, center_day, end_day


    def visualize(
        self, 
        cnames: list[str] | str, 
        date: str | datetime | None = None, 
        days_behind: int = 0,
        days_ahead: int | None = 0,
        figsize: tuple[int, int] = (12,4),
        grid: bool = True,
    ) -> np.ndarray:
        
        if date is None and days_behind > 0:
            raise ValueError

        if isinstance(cnames, str):
            cnames = [cnames]

        start_day, _, end_day = self._get_start_center_end_days(date=date, days_behind=days_behind, days_ahead=days_ahead)

        tmp = self.df.loc[start_day: end_day]
        tmp.index = pd.to_datetime(tmp.index)

        fig, axs = plt.subplots(len(cnames), 1, figsize=(figsize[0],figsize[1]*len(cnames)))
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for cname, ax in zip(cnames, axs):
            _ = ax.plot(tmp[cname].index, tmp[cname], color='tab:blue')
            aut_loc = mdates.AutoDateLocator()
            aut_fmt = mdates.ConciseDateFormatter(aut_loc)
            _ = ax.xaxis.set_major_locator(aut_loc)
            _ = ax.xaxis.set_major_formatter(aut_fmt)
            if grid:
                _ = ax.grid(axis='x')
            _ = ax.set_title(cname)
            
        fig.tight_layout()

        return axs