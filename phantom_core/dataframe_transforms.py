import datetime
from typing import TypeVar, overload
import pandas as pd
import numpy as np
from typing import Literal
import pandas as pd
from datetime import time
from pandas._libs.tslibs.offsets import Tick
from pydantic import BaseModel, field_validator

from .pydantic import pdTimestamp
from .constants import COL_LEVEL_SEP, DEFAULT_COLUMN_LEVEL_NAN, DATA_TIME_ZONE
from .datasource import time_data_table, DataTimeframe
from .market_dataframe import MarketDataFrame
from .market_calendar import get_market_days
from .utils import is_list_of_type


def convert_df_to_numeric(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Loops over columns and attempts to convert each one into numeric. If the conversion fails,
    it assumes this is not intended to be a numeric column and leaves it as-is.
    """
    for col in df.columns:
        try:
            # Attempt to convert to numeric
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            # If conversion fails, keep the original column
            if verbose:
                print(f'Found non-numeric column "{col}" in df')
            continue
    return df

DataFrameType = TypeVar('DataFrameType', pd.DataFrame, MarketDataFrame)

@overload
def drop_zero_variance_columns(df: pd.DataFrame) -> pd.DataFrame:
    ...

@overload
def drop_zero_variance_columns(df: MarketDataFrame) -> MarketDataFrame:
    ...

def drop_zero_variance_columns(df: DataFrameType) -> DataFrameType:
    """
    Remove columns with zero variance from the input DataFrame.

    Args:
        df (DataFrameType): Input DataFrame (either pd.DataFrame or MarketDataFrame).

    Returns:
        DataFrameType: DataFrame with zero variance columns removed.
    """
    no_variance_mask = df.std() == 0
    no_variance_cnames = no_variance_mask[no_variance_mask].index.tolist()
    df.drop(columns=no_variance_cnames, inplace=True)
    return df


def get_zero_variance_cols(df: DataFrameType) -> list[str]:
    no_variance_mask = df.std() == 0
    no_variance_cnames = no_variance_mask[no_variance_mask].index.tolist()
    assert is_list_of_type(no_variance_cnames, str)
    return no_variance_cnames


def get_cos_sin(col: pd.Series, num_periods: int) -> tuple[pd.Series, pd.Series]:
    """
    Calculate sine and cosine transformations of a given column.

    Args:
        col (pd.Series): Input column to transform.
        num_periods (int): Number of periods for the transformation.

    Returns:
        tuple[pd.Series, pd.Series]: Tuple containing sine and cosine transformed series.
    """
    sin = np.sin(col) * (2 * np.pi / num_periods)
    sin = pd.Series(sin)
    sin.name = str(col.name) + '_sin'
    cos = np.cos(col) * (2 * np.pi / num_periods)
    cos = pd.Series(cos)
    cos.name = str(col.name) + '_cos'
    return sin, cos


def extract_time_features(
        df: pd.DataFrame, 
        concat: bool = False, 
        format_cnames: bool = True, 
        force_intraday: bool = False,
        intraday_feats: list[str] | str = 'all'
    ) -> MarketDataFrame:
    """
    Extract various datetime features from the index of a DataFrame.

    This function adds new columns to the input DataFrame, including:
    - week: ISO calendar week number
    - month: Month number (1-12)
    - weekday: Day of the week (0-6, where 0 is Monday)
    - weekday_sin, weekday_cos: Sine and cosine transformations of weekday
    - month_sin, month_cos: Sine and cosine transformations of month

    For DataFrames with intraday frequency (less than 24 hours between entries), it also adds:
    - hour: Hour of the day (0-23)
    - minute: Minute of the hour (0-59)
    - period_num: Sequential number of the entry within its day
    - period_num_sin, period_num_cos: Sine and cosine transformations of period_num

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime index.
        concat (bool, optional): If True, adds new columns to the input DataFrame. 
                                 If False, creates a new DataFrame with only the extracted features. 
                                 Defaults to False.
        format_cnames (bool, optional): If True, formats the column names to include ticker and table information.
                                        Defaults to True.
        force_intraday (bool, optional): If True, forces the extraction of intraday features even if the
                                         DataFrame doesn't have an intraday frequency. Defaults to False.

    Returns:
        MarketDataFrame: DataFrame with additional columns for datetime features.

    Raises:
        AssertionError: If the input DataFrame's index is not a DatetimeIndex.
        ValueError: If intraday data has less than 2 periods per day.
    """
    
    assert isinstance(df.index, pd.DatetimeIndex)
    if not concat:
        df = pd.DataFrame(index=df.index)

    assert isinstance(df.index, pd.DatetimeIndex)

    incoming_cnames = df.columns.to_list()

    df['week'] = df.index.to_series().dt.isocalendar().week.astype(int)
    df['month'] = df.index.to_series().dt.month.astype(int)
    df['weekday'] = df.index.to_series().dt.weekday
    df['weekday_sin'] = np.sin(df['weekday'] * (2 * np.pi / 7))
    df['weekday_cos'] = np.cos(df['weekday'] * (2 * np.pi / 7))
    df['month_sin'] = np.sin(df['month'] * (2 * np.pi / 12))
    df['month_cos'] = np.cos(df['month'] * (2 * np.pi / 12))

    # Check if the index has intraday frequency
    if force_intraday or df.index.to_series().diff().median().total_seconds() < 24*60*60:  # Less than a day

        intra_df = pd.DataFrame(index=df.index)

        intra_df['_date'] = intra_df.index.normalize() # type: ignore
        num_periods_per_day = intra_df.groupby('_date').size().median()
        if num_periods_per_day < 2:
            raise ValueError(f"Intraday data must have at least 2 periods per day. Found a median of {num_periods_per_day}.")

        # Extract hour and minute
        intra_df['hour'] = intra_df.index.to_series().dt.hour
        intra_df['minute'] = intra_df.index.to_series().dt.minute

        num_periods_per_day = intra_df.groupby('_date').size().median()
        if num_periods_per_day < 2:
            raise ValueError(f"Intraday data must have at least 2 periods per day. Found a median of {num_periods_per_day}.")
        intra_df['period_num'] = intra_df.groupby('_date').cumcount()

        intra_df['period_num_sin'] = np.sin(intra_df['period_num'] * (2 * np.pi / num_periods_per_day))
        intra_df['period_num_cos'] = np.cos(intra_df['period_num'] * (2 * np.pi / num_periods_per_day))

        intra_df.drop(columns=['_date'], inplace=True)

        if isinstance(intraday_feats, list):
            intra_df = intra_df[intraday_feats]
        elif intraday_feats != 'all':
            raise ValueError(f"intraday_feats must be a list of column names or 'all', not {intraday_feats}")
        
        df = pd.concat([df, intra_df], axis=1)

    df = MarketDataFrame(df)

    if format_cnames:
        ticker = DEFAULT_COLUMN_LEVEL_NAN
        table = time_data_table.db_name

        new_cnames = [cname for cname in df.columns if cname not in incoming_cnames]
        new_cnames_formatted = [f'{ticker}{COL_LEVEL_SEP}{table}{COL_LEVEL_SEP}{cname}' for cname in new_cnames]
        df.columns = new_cnames_formatted
        
        df.set_col_level_names(names=[df.ticker_col_level_name, df.table_col_level_name, df.field_col_level_name])

    return df


def intersect_pandas_indexes(objs: list[pd.DataFrame | pd.Series]) -> list[pd.DataFrame | pd.Series]:
    """
    Intersect the indexes of multiple pandas objects and return objects with the common index.

    Args:
        objs (list[pd.DataFrame | pd.Series]): List of pandas objects to intersect.

    Returns:
        list[pd.DataFrame | pd.Series]: List of pandas objects with intersected index.

    Raises:
        AssertionError: If less than two objects are provided.
    """
    assert len(objs) >= 2, "At least two objects are required for intersection."
    common_index = objs[0].index
    for obj in objs[1:]:
        common_index = common_index.intersection(obj.index.to_list())
    out = [obj.loc[common_index].copy() for obj in objs]
    return out


def share_same_index(*args: pd.DataFrame | pd.Series) -> bool:
    """
    Check if all input pandas objects share the same index.

    Args:
        *args: Variable number of pandas DataFrames or Series.

    Returns:
        bool: True if all objects share the same index, False otherwise.
    """
    if len(args) < 2:
        return True
    
    first_index = args[0].index
    return all(obj.index.equals(first_index) for obj in args[1:])


def reindex_timeseries_df(
    df: pd.DataFrame, 
    freq: pd.Timedelta | DataTimeframe | datetime.timedelta | None = None, 
    start: pd.Timestamp | None = None, 
    end: pd.Timestamp | None = None, 
    start_end_inclusive: Literal['left', 'right', 'both', 'neither'] = 'both',
    between_time: tuple[time, time] | None = None, 
    between_time_inclusive: Literal['left', 'right', 'both', 'neither'] = 'both',
    respect_valid_market_days: bool = True,
) -> pd.DataFrame:
    """
    Reindex a time series DataFrame to a specified frequency and time range.

    Assumes a constant timezone throughout.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        freq (pd.Timedelta | DataTimeframe | datetime.timedelta | None): Desired 
            frequency for reindexing. If None, uses the DataFrame's index frequency.
        start (pd.Timestamp | None): Start timestamp for reindexing. If None, uses 
            the first timestamp in df.
        end (pd.Timestamp | None): End timestamp for reindexing. If None, uses the 
            last timestamp in df.
        start_end_inclusive (Literal['left', 'right', 'both', 'neither']): How to 
            handle inclusive intervals for start/end range.
        between_time (tuple[time, time] | None): Tuple of (start_time, end_time) 
            to filter timestamps within each day.
        between_time_inclusive (Literal['left', 'right', 'both', 'neither']): How 
            to handle inclusive intervals for between_time filtering.
        respect_valid_market_days (bool): If True, only include valid market days 
            in the reindexed DataFrame.

    Returns:
        pd.DataFrame: Reindexed DataFrame.

    Raises:
        ValueError: If frequency or index requirements are not met.
        AssertionError: If the input DataFrame's index is not a DatetimeIndex.

    Notes:
        - The function assumes that the input DataFrame has a DatetimeIndex.
        - If no frequency is provided and the DataFrame's index doesn't have a 
          built-in frequency, a ValueError is raised.
        - For daily frequency (freq=pd.Timedelta(days=1)), between_time must be 
          None and the index must contain only normalized timestamps.
        - The function ensures that the index is sorted in ascending order.
        - If respect_valid_market_days is True, only timestamps corresponding to 
          valid market days are included in the output.
    """
    
    assert isinstance(df.index, pd.DatetimeIndex)

    # Determine the frequency to use
    if freq is None:
        if df.index.freq is None:
            raise ValueError('did not pass a freq and index does not have one built in')
        assert isinstance(df.index.freq, Tick)
        _freq = df.index.freq.delta
    else:
        freq = pd.Timedelta(freq)
        if df.index.freq is not None:
            assert isinstance(df.index.freq, Tick)
            if pd.Timedelta(df.index.freq.delta) != freq:
                raise ValueError(f'df index has a freq of {df.index.freq.delta}, not {freq}')
        _freq = freq
    
    # Check for daily frequency constraints
    if _freq >= pd.Timedelta(days=1):
        if between_time is not None:
            raise ValueError('between_time must be None for daily freq')
        if not (df.index == df.index.normalize()).all():
            raise ValueError('daily df must have DatetimeIndex with only normalized timestamps')

    # Ensure the index is sorted
    if not df.index.is_monotonic_increasing:
        raise ValueError('df DateTimeIndex is not sorted')

    # Set start and end if not provided
    start = start if start is not None else df.index[0]
    end = end if end is not None else df.index[-1]

    # Validate timezone consistency
    if start.tzinfo is None:
        raise ValueError("start timestamp must have timezone info (cannot be tz-naive)")
    if end.tzinfo is None:
        raise ValueError("end timestamp must have timezone info (cannot be tz-naive)")
    if df.index.tzinfo is None:
        raise ValueError("DataFrame index must have timezone info (cannot be tz-naive)")
    if str(start.tzinfo) != str(df.index.tzinfo):
        raise ValueError(
            f"start timestamp timezone ({start.tzname()}) does not match "
            f"DataFrame index timezone ({df.index.tzinfo.tzname(None)})"
        )
    if str(end.tzinfo) != str(df.index.tzinfo):
        raise ValueError(
            f"end timestamp timezone ({end.tzname()}) does not match "
            f"DataFrame index timezone ({df.index.tzinfo.tzname(None)})"
        )
    # Create a new date range
    periods = pd.date_range(start=start, end=end, freq=_freq, inclusive=start_end_inclusive)

    # Apply between_time filter if specified
    if between_time is not None:
        start_time, end_time = between_time
        periods = periods.to_series().between_time(
            start_time=start_time, 
            end_time=end_time, 
            inclusive=between_time_inclusive # type: ignore
        ).index 
        assert isinstance(periods, pd.DatetimeIndex)

    # Reindex the DataFrame
    df = df.reindex(periods)

    if respect_valid_market_days:
        assert isinstance(df.index, pd.DatetimeIndex)
        df['_date'] = df.index.normalize()
        start_date = df['_date'].iloc[0]
        end_date = df['_date'].iloc[-1]
        market_days = get_market_days(start_ts=start_date, end_ts=end_date)
        df = df.loc[df['_date'].isin(market_days)]
        df.drop(columns=['_date'], inplace=True)

    return df


def add_null_row_for_timestamp(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """
    Add a new row with null values for a specific timestamp to a DataFrame.

    This function creates a new row with null values for all columns in the input DataFrame
    at the specified timestamp. The new row is then concatenated with the original DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to which the null row will be added.
        ts (pd.Timestamp): The timestamp for the new null row.

    Returns:
        pd.DataFrame: A new DataFrame with the null row added at the specified timestamp.

    Note:
        The function preserves the data types of the original DataFrame columns in the new null row.
    """

    null_row = pd.DataFrame(index=[ts], columns=df.columns, dtype=object)
    null_row[:] = None
    
    for col in df.columns:
        null_row[col] = null_row[col].astype(df[col].dtype)

    return pd.concat([df, null_row])


class DateFilter(BaseModel):
    
    start: pdTimestamp | None = None
    end: pdTimestamp | None = None
    start_inclusive: bool = True
    end_inclusive: bool = True


    @field_validator('start', 'end')
    def validate_normalized_timestamp(cls, v):
        if v is not None and v.normalize() != v:
            raise ValueError('Timestamp must be normalized (set to midnight)')
        return v


    @overload
    def filter(self, obj: pd.DataFrame) -> pd.DataFrame: ...
    @overload
    def filter(self, obj: MarketDataFrame) -> MarketDataFrame: ...
    @overload
    def filter(self, obj: pd.Series) -> pd.Series: ...
    def filter(self, obj: pd.DataFrame | MarketDataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    
        assert isinstance(obj.index, pd.DatetimeIndex)

        start_mask = pd.Series(True, index=obj.index)
        end_mask = pd.Series(True, index=obj.index)

        if self.start is not None:
            if self.start_inclusive:
                start_mask = obj.index.normalize() >= self.start
            else:
                start_mask = obj.index.normalize() > self.start
        
        if self.end is not None:
            if self.end_inclusive:
                end_mask = obj.index.normalize() <= self.end
            else:
                end_mask = obj.index.normalize() < self.end

        return obj.loc[start_mask & end_mask]


class TimeFilter(BaseModel):
    start: datetime.time
    end: datetime.time
    inclusive: str = 'both'

    @field_validator('inclusive')
    def validate_inclusive(cls, v):
        if v not in {'both', 'neither', 'left', 'right'}:
            raise ValueError("inclusive must be one of 'both', 'neither', 'left', or 'right'")
        return v

    @overload
    def filter(self, obj: pd.DataFrame) -> pd.DataFrame: ...
    @overload
    def filter(self, obj: MarketDataFrame) -> MarketDataFrame: ...
    @overload
    def filter(self, obj: pd.Series) -> pd.Series: ...
    def filter(self, obj: pd.DataFrame | MarketDataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        assert isinstance(obj.index, pd.DatetimeIndex)
        return obj.between_time(start_time=self.start, end_time=self.end, inclusive=self.inclusive) # type: ignore


def copy_constant_col_to_all_rows(df: pd.DataFrame, cname: str) -> pd.DataFrame:
    """
    Copy a constant column to all rows, allowing for missing values in the column as long
    as the there is only one unique non-missing value in the column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cname (str): Name of the column to copy.

    Returns:
        pd.DataFrame: DataFrame with the constant column copied to all rows.
    """

    if cname not in df.columns:
        raise ValueError(f'{cname} not in df.columns')

    unique_vals = set(df[cname].dropna().unique())

    if len(unique_vals) > 1:
        raise ValueError(f'{cname} has more than one unique value')
    if len(unique_vals) == 0:
        raise ValueError(f'{cname} has no unique values')

    df.loc[:, cname] = unique_vals.pop()

    return df