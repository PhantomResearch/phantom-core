from enum import Enum
import inspect
from pathlib import Path
import time
import pandas as pd
from typing import Any, Literal, Tuple, Type, TypeGuard, TypeVar, overload
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np
from logging import Logger
import os
from contextlib import contextmanager
import datetime

from .logging import get_logger
from .constants import DEFAULT_DATE_FORMAT_STR



def configure_pandas_display(max_rows: int = 200, max_columns: int = 200):
    """
    Helper function to set pandas display properties in a notebook.

    Args:
        max_rows (int): Maximum number of rows to display. Defaults to 200.
        max_columns (int): Maximum number of columns to display. Defaults to 200.
    """
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_colwidth', 1000)



def plot_dataframe_nulls(df: pd.DataFrame, figsize: tuple[float, float] | None = None, max_rows: int = 5000) -> Figure:
    """
    Plot a heatmap of missing values in a DataFrame.

    This function creates a heatmap visualization of missing values in the input DataFrame.
    The heatmap uses blue for present values and red for missing values. It includes a legend
    and adjusts the x-axis labels for better readability.

    Args:
        df (pd.DataFrame): Input DataFrame to visualize.
        figsize (tuple[float, float] | None): Figure size (width, height) in inches. 
            If None, it's calculated based on the DataFrame dimensions.
        max_rows (int): Maximum number of rows to display. Defaults to 5000.
            Note: This parameter is currently not used in the function.

    Returns:
        Figure: A matplotlib Figure object containing the heatmap visualization.

    Note:
        - The function uses seaborn's heatmap for visualization.
        - X-axis labels are rotated and spaced for better readability.
        - A legend is added to distinguish between present and missing values.
        - The y-axis shows column names of the DataFrame.
    """
    missing_mask = df.isna()

    fig_height = len(df.columns) * 0.18
    fig_width = 12
    figsize = figsize if figsize else (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(missing_mask.T, cmap=['#6495ED', '#B22222'], ax=ax, cbar=False)

    ax.set_title('Heatmap of Missing Values')
    ax.set_xlabel('Date')
    ax.set_ylabel('Columns')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    n = len(df.index) // 10
    ax.set_xticks(range(0, len(df.index), n))
    ax.set_xticklabels(df.index[::n])

    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns, rotation=0)

    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

    present_patch = mpatches.Patch(color='#6495ED', label='Present')
    missing_patch = mpatches.Patch(color='#B22222', label='Missing')
    ax.legend(handles=[present_patch, missing_patch], loc='upper right', bbox_to_anchor=(1.1, 1))

    fig.tight_layout()

    return fig


def get_combined_null_mask(df: pd.DataFrame, combination_type: str) -> pd.Series:
    """
    Get a combined mask of null values in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        combination_type (str): Type of combination, either 'any' or 'all'.

    Returns:
        pd.Series: A boolean Series representing the combined null mask.

    Raises:
        AssertionError: If combination_type is not 'any' or 'all'.
    """
    assert combination_type in ('any', 'all')
    if combination_type == 'any':
        return df.isna().any(axis=1)
    elif combination_type == 'all':
        return df.isna().all(axis=1)


def get_today(as_type: str = 'datetime') -> datetime.datetime | str | pd.Timestamp:
    """
    Get today's date in the specified type.

    Args:
        as_type (str): The desired return type. Must be one of 'datetime', 'str', or 'pd.timestamp'. Defaults to 'datetime'.

    Returns:
        datetime | str | pd.Timestamp: Today's date in the specified format.

    Raises:
        AssertionError: If as_type is not one of the allowed values.
    """
    as_type = as_type.lower()
    assert as_type in ('datetime', 'str', 'pd.timestamp')

    pd_now = pd.Timestamp.now().normalize()

    if as_type == 'pd.timestamp':
        return pd_now
    elif as_type == 'datetime':
        return pd_now.to_pydatetime()
    else:
        return pd_now.to_pydatetime().strftime(DEFAULT_DATE_FORMAT_STR)


def get_toy_df() -> pd.DataFrame:
    """
    Create a toy DataFrame with various date formats and some null values.

    Returns:
        pd.DataFrame: A DataFrame with sample data.
    """
    dates_str = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    dates_dt = [datetime.datetime.strptime(d, DEFAULT_DATE_FORMAT_STR) for d in dates_str]
    dates_date = [dt.date() for dt in dates_dt]
    dates_pts = pd.Series(pd.to_datetime(dates_str))
    return pd.DataFrame({
        'date_str': dates_str,
        'date_dt': dates_dt,
        'date_pts': dates_pts,
        'date_date': dates_date,
        'date': dates_dt,
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, np.nan, 3, 1],
        'C': [np.nan, 2, 3, np.nan, 5],
        'D': ['a', 'b', 'c', 'd', np.nan]
    })


def get_toy_df_multiindex() -> pd.DataFrame:
    """
    Create a toy DataFrame with a MultiIndex.

    Returns:
        pd.DataFrame: A DataFrame with a MultiIndex and random data.
    """
    columns = pd.MultiIndex.from_product([['1', '2'], ['a', 'b', 'c']], 
                                        names=['Level1', 'Level2'])
    return pd.DataFrame(np.random.rand(3, 6), columns=columns)


def toggle_date_type(date: str | datetime.datetime, format: str | None = None) -> str | datetime.datetime:
    """
    Toggle between string and datetime.datetime format.

    Args:
        date (str | datetime): The date to toggle.
        format (str | None): The date format string. Defaults to None, which uses DEFAULT_DATE_FORMAT_STR.

    Returns:
        str | datetime: The toggled version of the input date.

    Raises:
        ValueError: If the input date is neither a string nor a datetime object.
    """
    format = format if format else DEFAULT_DATE_FORMAT_STR
    if isinstance(date, datetime.datetime):
        return date.strftime(format)
    elif isinstance(date, str):
        return datetime.datetime.strptime(date, format)
    else:
        raise ValueError('date has wrong format')


def datetime_ify(date: str | datetime.datetime) -> datetime.datetime:
    """
    Coerce an object into a datetime.datetime object.

    Args:
        date (str | datetime): The date to coerce.

    Returns:
        datetime: The coerced date.
    """
    if isinstance(date, str):
        date = toggle_date_type(date)
        assert isinstance(date, datetime.datetime)
        return date
    return date


def filter_cnames(df: pd.DataFrame, *filters, how: str = 'and') -> list[str]:
    """
    Filter dataframe column names based on provided filters.

    Args:
        df (pd.DataFrame): DataFrame for which you want to filter column names.
        *filters: Variable number of strings to use as filters.
        how (str): Specifies whether it's an "or" filter or an "and" filter. Must be 'and' or 'or'.

    Returns:
        list[str]: A list of columns matching the filter.

    Raises:
        AssertionError: If 'how' is not 'and' or 'or'.
    """
    assert how in ('and', 'or')

    how_func = {'and': all, 'or': any}

    results = []

    for cname in df.columns:
        filter_results: list[bool] = [x.lower() in cname.lower() for x in filters]

        if how_func[how](filter_results):
            results.append(cname)

    return results


def filter_cnames_mi(df: pd.DataFrame, f: str | list[str]) -> list[tuple]:
    """
    Filter column names of a DataFrame with MultiIndex columns.

    Args:
        df (pd.DataFrame): DataFrame with MultiIndex columns.
        f (str | list[str]): Filter string or list of filter strings.

    Returns:
        list[tuple]: List of column names (as tuples) that match the filter.
    """
    if not isinstance(f, list):
        f = [f]
    results = []
    for cname in df.columns:
        if all([x.lower() in '_'.join(list(cname)).lower() for x in f]):
            results.append(cname)
    return results


def toggle_index(df: pd.DataFrame | pd.Series, to: str | None = None) -> None:
    """
    Toggle the index of a DataFrame or Series between DatetimeIndex and RangeIndex.

    Args:
        df (pd.DataFrame | pd.Series): The DataFrame or Series to modify.
        to (str | None): The desired index type ('range' or 'dt'). If None, it toggles to the opposite type.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        if not to or to=='range':
            df.reset_index(inplace=True)
    elif isinstance(df.index, pd.RangeIndex):
        if not to or to=='dt':
            df.set_index('date', inplace=True)
    print(f'df now has a {type(df.index).__name__} index')


def get_first_and_last_valid_index(series: pd.Series) -> Tuple:
    """
    Return the index values for the first non-null value and the last non-null value.

    Args:
        series (pd.Series): Input series.

    Returns:
        Tuple: A tuple containing the first and last valid index.
    """
    return series.first_valid_index(), series.last_valid_index()


def collapse_column_levels(df_cols: pd.MultiIndex, n: int | None = None, sep: str = '_') -> pd.Index:
    """
    Collapse n levels bottom to top of a MultiIndex. Collapses all levels by default if n is not passed.

    Args:
        df_cols (pd.MultiIndex): The multi-level columns of a dataframe.
        n (int | None): What level from the bottom to collapse to, starting at 1. Defaults to None.
        sep (str): String separator for column levels in the collapsed representation. Defaults to '_'.

    Returns:
        pd.Index: The collapsed index.
    """
    if not isinstance(df_cols, pd.MultiIndex) or n == 1:
        return df_cols

    existing_cols = df_cols.copy()
    n_existing_levels = existing_cols.nlevels
    n_ = n if n else n_existing_levels
    n_ = min(n_, n_existing_levels)
    
    if n_ == n_existing_levels:
        new_index = df_cols.map(lambda x: sep.join(str(i) for i in x if pd.notna(i)))
        new_index.names = [existing_cols.names[0]]
    else:
        def collpase_levels(tup):
            upper_levels = tup[:-n_]
            lower_levels = '_'.join(str(i) for i in tup[-n_:] if pd.notna(i))
            return upper_levels + (lower_levels,)
        
        new_columns = map(collpase_levels, existing_cols.to_list())
        new_names = list(existing_cols.names[:-n_]) + [sep.join(str(name) for name in existing_cols.names[-n_:] if pd.notna(name))]
        new_index = pd.MultiIndex.from_tuples(new_columns, names=new_names)
    
    return new_index


def is_function(obj: Any) -> bool:
    """
    Check if an object is a function or method.

    Args:
        obj (Any): The object to check.

    Returns:
        bool: True if the object is a function or method, False otherwise.
    """
    return callable(obj)


def function_to_serializable(func: Any) -> dict | Any:
    """
    Convert a function to a serializable format.

    Args:
        func (Any): The function to convert.

    Returns:
        dict | Any: A dictionary representation of the function if it's a function, otherwise returns the input unchanged.
    """
    if is_function(func):
        return {
            'type': 'function',
            'name': func.__name__,
            'source': inspect.getsource(func)
        }
    return func


def recursively_serialize_dict(obj: Any) -> Any:
    """
    Recursively serialize a dictionary, converting functions to serializable format.

    Args:
        obj (Any): The object to serialize.

    Returns:
        Any: The serialized object.
    """
    if isinstance(obj, dict):
        return {k: recursively_serialize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursively_serialize_dict(item) for item in obj]
    elif is_function(obj):
        return function_to_serializable(obj)
    else:
        return obj


def delete_files_in_dir(dir: str | Path) -> None:
    """
    Delete all files in a directory.

    Args:
        dir (str | Path): The directory path.
    """
    for fp in Path(dir).glob('*'):
        if fp.is_file():
            fp.unlink()


def display_countdown(
        msg: str, 
        num_seconds: int = 10, 
        logger: Logger | None = None, 
        level: Literal['info', 'warning'] = 'warning',
        display_updates: bool = True,
    ) -> None:
    """
    Display a countdown message with a specified delay.

    This function logs a countdown message at regular intervals, using either a provided logger
    or creating a new one if not provided. The message is logged at the specified level (info or warning).

    Args:
        msg (str): The base message to display in the countdown.
        num_seconds (int): The total number of seconds to count down from.
        logger (Logger | None): The logger to use for output. If None, a new logger is created using the current module name.
        level (Literal['info', 'warning']): The logging level to use. Defaults to 'warning'.
        display_updates (bool): Whether to display updates to the console. Defaults to True.
    Returns:
        None

    Example:
        >>> display_countdown("Starting process", 5, level='info')
        # This will log "Starting process in 5 seconds..." and count down to 1 second.
    """
    if logger is None:
        logger = get_logger(name=__name__)

    def _log(msg, seconds_remaining):
        _msg = msg + f' (sleeping for {seconds_remaining} seconds...)'
        if level == 'warning':
            logger.warning(_msg)
        else:
            logger.info(_msg)

    _log(msg, num_seconds)

    for seconds_remaining in range(num_seconds, 0, -1):

        time.sleep(1)

        if display_updates:
            _log(msg, seconds_remaining-1)
        


T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

def is_list_of_type(var: Any, expected_type: Type[T]) -> TypeGuard[list[T]]:
    """
    Check if a variable is a list of a specific type.

    This function uses a TypeGuard to provide type narrowing, allowing the type checker
    to recognize the variable as a list of the expected type if the function returns True.

    Args:
        var (Any): The variable to check.
        expected_type (Type[T]): The expected type of the list items.

    Returns:
        TypeGuard[list[T]]: True if var is a list and all items are of expected_type, False otherwise.

    Example:
        >>> numbers = [1, 2, 3]
        >>> if is_list_of_type(numbers, int):
        ...     # Type checker knows that numbers is List[int] here
        ...     total = sum(numbers)
    """
    return isinstance(var, list) and all(type(item) is expected_type for item in var)


def is_dict_of_type(var: Any, expected_key_type: Type[K], expected_value_type: Type[V]) -> TypeGuard[dict[K, V]]:
    """
    Check if a variable is a dictionary with specific key and value types.

    This function uses a TypeGuard to provide type narrowing, allowing the type checker
    to recognize the variable as a dictionary of the expected key and value types if the function returns True.

    Args:
        var (Any): The variable to check.
        expected_key_type (Type[K]): The expected type of the dictionary keys.
        expected_value_type (Type[V]): The expected type of the dictionary values.

    Returns:
        TypeGuard[dict[K, V]]: True if var is a dict with keys of expected_key_type and values of expected_value_type, False otherwise.

    Example:
        >>> person = {"name": "Alice", "age": 30}
        >>> if is_dict_of_type(person, str, int):
        ...     # Type checker knows that person is Dict[str, int] here
        ...     age = person["age"] + 1
    """
    return isinstance(var, dict) and all(type(k) is expected_key_type and type(v) is expected_value_type for k, v in var.items())


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flatten a nested dictionary into a single-level dictionary.

    This function recursively flattens a nested dictionary, joining keys with a separator.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): The parent key for nested dictionaries. Used to maintain the 
            hierarchy of the original nested structure in the flattened keys. Defaults to an empty string.
        sep (str, optional): The separator to use between keys. Defaults to '.'.

    Returns:
        dict: A flattened dictionary where nested keys are joined with the separator.

    Example:
        >>> nested_dict = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        >>> flatten_dict(nested_dict)
        {'a': 1, 'b.c': 2, 'b.d.e': 3}

    Note:
        - The function preserves the original values of the dictionary.
        - If a value in the dictionary is itself a dictionary, it will be flattened recursively.
        - The order of keys in the flattened dictionary may not be the same as in the original nested dictionary.
        - The `parent_key` parameter is crucial for the recursive process, building up the path to each nested element.
          It ensures unique keys in the flattened dictionary and allows for potential reconstruction of the original structure.
        - You can customize the initial `parent_key` to add a prefix to all flattened keys, which can be useful for
          namespacing or distinguishing between multiple flattened dictionaries.

    Advanced Usage:
        To add a namespace to all keys:
        >>> config = {'a': 1, 'b': {'c': 2}}
        >>> flatten_dict(config, parent_key='config')
        {'config.a': 1, 'config.b.c': 2}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


T = TypeVar('T')
@overload
def listify(item: list[T]) -> list[T]: ...
@overload
def listify(item: T) -> list[T]: ...

def listify(item: T | list[T]) -> list[T]:
    """
    Convert an item to a list if it's not already a list.

    This function is useful when you want to ensure that you're working with a list,
    regardless of whether the input is a single item or already a list.

    Args:
        item (T | List[T]): The item to be converted to a list.

    Returns:
        List[T]: A list containing the input item if it wasn't already a list,
                 or the original list if the input was already a list.

    Examples:
        >>> listify(1)
        [1]
        >>> listify([1, 2, 3])
        [1, 2, 3]
        >>> listify("hello")
        ['hello']
        >>> listify(None)
        [None]
    """

    if isinstance(item, list):
        return item
    return [item]


def get_first_nonnull_ts(df: pd.DataFrame, how: Literal['any', 'all'] = 'any', cname_subset: list[str] | None = None) -> pd.Timestamp:
    """
    Get the first timestamp where columns in the DataFrame have non-null values based on the specified condition.

    This function finds the earliest timestamp in the DataFrame's index where the specified condition
    ('any' or 'all') is met for non-null values across columns. It assumes that the DataFrame has a
    DatetimeIndex that is sorted in ascending order.

    Args:
        df (pd.DataFrame): The input DataFrame with a DatetimeIndex.
        how (Literal['any', 'all'], optional): Specifies the condition for non-null values.
            'any': Return the first timestamp where any column has a non-null value.
            'all': Return the first timestamp where all columns have non-null values.
            Defaults to 'any'.
        cname_subset (list[str] | None, optional): A list of column names to consider. If None, all columns are considered. Defaults to None.
    Returns:
        pd.Timestamp: The first timestamp where the specified condition is met for non-null values.

    Raises:
        AssertionError: If the DataFrame's index is not a DatetimeIndex or if the
                        first non-null timestamp is not a pd.Timestamp object.
        ValueError: If the DataFrame's DatetimeIndex is not sorted in ascending order,
                    or if an invalid 'how' parameter is provided.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'A': [None, 1, 2],
        ...     'B': [None, None, 3]
        ... }, index=pd.date_range('2023-01-01', periods=3))
        >>> get_first_nonnull_ts(df)
        Timestamp('2023-01-02 00:00:00')
        >>> get_first_nonnull_ts(df, how='all')
        Timestamp('2023-01-03 00:00:00')
    """
    assert isinstance(df.index, pd.DatetimeIndex)
    if not df.index.is_monotonic_increasing:
        raise ValueError('df DatetimeIndex is not sorted')
    
    if cname_subset is not None:
        _df = df.loc[:, cname_subset]
    else:
        _df = df
    
    if how == 'any':
        nonnull_mask = _df.notnull().any(axis=1)
    elif how == 'all':
        nonnull_mask = _df.notnull().all(axis=1)
    else:
        raise ValueError(f'Invalid how: {how}')
    
    first_nonnull_ts = nonnull_mask[nonnull_mask].index[0]
    assert isinstance(first_nonnull_ts, pd.Timestamp)
    return first_nonnull_ts


@contextmanager
def working_directory(path):
    """
    A context manager that temporarily changes the working directory.

    Args:
        path (str): The path to the directory to change to.

    Yields:
        None
    """
    current_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current_dir)


def subtract_timedelta_from_time(t: datetime.time, delta: pd.Timedelta) -> datetime.time:
    """
    Subtract a timedelta from a datetime.time object.
    
    Args:
        t (time): The time to subtract from.
        delta (timedelta): The timedelta to subtract.
    
    Returns:
        time: The resulting time after subtraction.
    """
    # Convert time to datetime
    dt = datetime.datetime.combine(datetime.datetime.min, t)
    
    # Subtract timedelta
    result = dt - delta
    
    # Return the time component
    return result.time()


def coerce_to_tstz(ts: pd.Timestamp | datetime.datetime | str, default_ts: str = 'US/Eastern') -> pd.Timestamp:
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    if isinstance(ts, datetime.datetime):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(default_ts)
    return ts.tz_convert(default_ts)


def coerce_to_tstz_str(ts: pd.Timestamp | datetime.datetime | str, default_ts: str = 'US/Eastern') -> str:
    return coerce_to_tstz(ts, default_ts).strftime('%Y-%m-%d %H:%M:%S.%f %Z')


EnumT = TypeVar('EnumT', bound=Enum)
def coerce_to_enum(s: str | EnumT, enum_type: type[EnumT], coerce: bool = True) -> EnumT:
    if isinstance(s, Enum):
        return s

    if not coerce:
        return enum_type(s)
    
    enum_vals = [e.value for e in enum_type]
    match_idx = None
    for i, val in enumerate(enum_vals):
        if s.lower() == val.lower():
            match_idx = i
            break
    
    if match_idx is None:
        raise ValueError(f"Invalid {enum_type.__name__}: {s}")
    
    return enum_type(enum_vals[match_idx])