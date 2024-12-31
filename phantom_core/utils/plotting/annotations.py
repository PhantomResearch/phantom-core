from matplotlib.axes import Axes
import matplotlib.dates as mdates



def annotate_ax(ax: Axes, title: str, xlabel: str | None = None, ylabel: str | None = None, grid: bool = True) -> Axes:
    _ = ax.set_title(title)
    if xlabel is not None:
        _ = ax.set_xlabel(xlabel)
    if ylabel is not None:
        _ = ax.set_ylabel(ylabel)
    if grid:
        _ = ax.grid()
    return ax


def format_time_axis_labels(ax: Axes) -> Axes:
    auto_loc = mdates.AutoDateLocator()
    auto_fmt = mdates.ConciseDateFormatter(auto_loc)
    _ = ax.xaxis.set_major_locator(auto_loc)
    _ = ax.xaxis.set_major_formatter(auto_fmt)
    return ax
