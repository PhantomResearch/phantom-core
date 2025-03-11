import pandas as pd
import datetime as dt
import pandas_market_calendars as mcal
from typing import Literal, overload
from datetime import time
import datetime

from .constants import NYSE_CALENDAR, DATA_TIME_ZONE, DEFAULT_COLUMN_LEVEL_NAN
from .datasource import DataTimeframe



def get_market_calendar(start_date: str = '2010-01-01', n_future_days: int = 365) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """
    Get an empty calendar in the form of a daily dataframe with trading days only.
    This calendar will span from `start_date` to `datetime.now() + n_future_days`.

    Returns:
        mkt_df: dataframe of days when the market is open, including holidays and weekends
        market_closed_list: a list of business days when the market isn't open (assuming these are holidays)
    """

    # TODO: is this still needed? Might be nice to keep around for future use, but clean it up

    # NYSE Stock Market Days Open
    nyse = mcal.get_calendar('NYSE')

    # Add 365 days to the current date
    next_year_date = dt.datetime.now() + dt.timedelta(days=n_future_days)

    mkt_df = nyse.schedule(start_date=f'{start_date}', end_date=f'{next_year_date}')
    mkt_df.index = pd.to_datetime(mkt_df.index).normalize()
    mkt_df['index_backup'] = mkt_df.index
    mkt_df['mkt_df_creation_filler'] = True

    # Identify all business days in the date range of mkt_df
    all_business_days = pd.date_range(start=mkt_df.index.min(), end=mkt_df.index.max(), freq='B')

    # Find missing business days (bank holidays)
    market_closed_days = all_business_days.difference(mkt_df.index.tolist())

    # Convert missing_days to a list
    market_closed_list = market_closed_days.to_list()

    mkt_df.sort_index(inplace=True)

    mkt_df.columns = pd.MultiIndex.from_tuples([(c, DEFAULT_COLUMN_LEVEL_NAN, DEFAULT_COLUMN_LEVEL_NAN, DEFAULT_COLUMN_LEVEL_NAN) for c in mkt_df.columns])

    return mkt_df.copy(), market_closed_list


def get_market_days(
    start_ts: pd.Timestamp, 
    end_ts: pd.Timestamp, 
    tz: str | None = None,
    # strip_tzinfo: bool = True            TODO
) -> list[pd.Timestamp]:
    """
    Get a list of market days between the given start and end timestamps.

    This function returns a list of Timestamps representing trading days for the NYSE
    between the specified start and end timestamps. The returned Timestamps will be normalized
    to midnight in the timezone of the input timestamps.

    Args:
        start_ts (pd.Timestamp): The start timestamp with timezone information.
        end_ts (pd.Timestamp): The end timestamp with timezone information.
        strip_tzinfo (bool): If True, remove timezone information from returned Timestamps. Defaults to True.   TODO

    Returns:
        list[pd.Timestamp]: A list of Timestamps representing market days, normalized
        to midnight in the input timezone. If strip_tzinfo is True, timezone information is removed.

    Raises:
        ValueError: If start_ts or end_ts do not have timezone information.
        ValueError: If start_ts and end_ts have different timezone information.

    Note:
        - User can provide tz-naive start and end ts with a tz, or tz-aware start and end ts without a tz
        - If strip_tzinfo is True, the returned Timestamps will have no timezone information (naive).          TODO
        - If strip_tzinfo is False, the returned Timestamps will retain the timezone of the input timestamps.  TODO
    """

    if tz is not None:
        if start_ts.tzinfo is not None or end_ts.tzinfo is not None:
            raise ValueError('cannot pass tz-aware start or end ts with tz arg')
        _tz = tz
    else:
        if start_ts.tzinfo is None or end_ts.tzinfo is None:
            raise ValueError('start_ts and end_ts must have timezone info if not passing tz arg')
        if str(start_ts.tzinfo) != str(end_ts.tzinfo):
            raise ValueError('start_ts and end_ts must have the same timezone info')
        _tz = str(start_ts.tzinfo)
        start_ts = start_ts.tz_localize(None)
        end_ts = end_ts.tz_localize(None)
        
    start_ts = start_ts.normalize()
    end_ts = end_ts.normalize()
    
    days = NYSE_CALENDAR.valid_days(start_date=start_ts, end_date=end_ts, tz=_tz).tolist()

    # if strip_tzinfo:
    #     days = [ts.tz_localize(None) for ts in days]          TODO

    return days


def ts_in_valid_market_day(ts: pd.Timestamp) -> bool:
    """
    Check if the given timestamp falls on a valid market day.

    This function determines whether the provided timestamp is on a day when the market is open.
    If the input timestamp doesn't have timezone information, it's localized to the DATA_TIME_ZONE.

    Args:
        ts (pd.Timestamp): The timestamp to check.

    Returns:
        bool: True if the timestamp is on a valid market day, False otherwise.

    Note:
        This function uses the get_market_days() function to determine valid market days.
        A valid market day is one where get_market_days() returns a non-empty list.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize(DATA_TIME_ZONE)
    return len(get_market_days(start_ts=ts, end_ts=ts)) > 0


class MarketTimestampMagic:
    """
    A generator class for creating market timestamps based on specified parameters.

    This class generates timestamps for market data, respecting market days and time ranges.
    It can be used as an iterator or accessed using indexing.

    Attributes:
        start_ts (pd.Timestamp): The starting timestamp for generation.
        freq (pd.Timedelta | DataTimeframe): The frequency of timestamp generation.
        between_time (tuple[time, time] | None): Optional time range to consider.
        inclusive (Literal['left', 'right', 'both', 'neither']): How to treat the time range boundaries.
        respect_valid_market_days (bool): Whether to only generate timestamps on valid market days.

    Usage:
        # Create a MarketTimestampMagic instance
        mtm = MarketTimestampMagic(start_ts=pd.Timestamp('2023-01-01', tz='UTC'),
                                   freq=pd.Timedelta(minutes=5),
                                   between_time=(pd.Timestamp('09:30').time(), pd.Timestamp('16:00').time()))

        # Use as an iterator (will be an infinite loop)
        for ts in mtm:
            print(ts)

        # Use indexing (primary use case)
        first_ts = mtm[0]
        next_10_ts = mtm[1:11]
    """

    def __init__(
        self,
        start_ts: pd.Timestamp,
        freq: pd.Timedelta | DataTimeframe, 
        between_time: tuple[time, time] | None = None, 
        between_time_inclusive: Literal['left', 'right', 'both', 'neither'] = 'both',
        respect_valid_market_days: bool = True,
        strip_tzinfo: bool = True,
    ):
        """
        Initialize the MarketTimestampGenerator.

        Args:
            start_ts (pd.Timestamp): The starting timestamp.
            freq (pd.Timedelta | DataTimeframe): The frequency of timestamp generation.
            between_time (tuple[time, time] | None): Optional time range to consider.
            inclusive (Literal['left', 'right', 'both', 'neither']): How to treat the time range boundaries.
            respect_valid_market_days (bool): Whether to only generate timestamps on valid market days.

        Raises:
            ValueError: If start_ts doesn't have timezone information or if it's not a valid timestamp.
        """
        if start_ts.tzinfo is None:
            raise ValueError('start_ts must have timezone info')
        
        if freq >= pd.Timedelta(days=1) and between_time is not None:
            raise ValueError('between_time cannot be provided if freq is daily or longer')
        
        self._tzinfo = start_ts.tzinfo
        
        self.start_ts = start_ts
        self.freq = freq
        self.between_time = between_time
        self.inclusive = between_time_inclusive
        self.respect_valid_market_days = respect_valid_market_days
        self.strip_tzinfo = strip_tzinfo

        self._current_ts = start_ts

        # If respecting market days, pre-fetch a buffer of valid market days
        if self.respect_valid_market_days:
            self._valid_market_days_buffer = get_market_days(start_ts=start_ts, end_ts=start_ts + pd.Timedelta(days=400))

        if not self._ts_valid(start_ts):
            raise ValueError('provided start_ts must be a valid timestamp')


    def _ts_in_valid_market_days(self, ts: pd.Timestamp) -> bool:
        """
        Check if the given timestamp is on a valid market day.

        Args:
            ts (pd.Timestamp): The timestamp to check.

        Returns:
            bool: True if the timestamp is on a valid market day, False otherwise.
        """
        if not self.respect_valid_market_days:
            return True
        
        if ts < self._valid_market_days_buffer[0]:
            raise ValueError(f'provided ts is before the valid market days buffer: {ts} < {self._valid_market_days_buffer[0]}')
        
        if ts.normalize() > self._valid_market_days_buffer[-1]:
            self._valid_market_days_buffer = get_market_days(start_ts=self.start_ts, end_ts=ts + pd.Timedelta(days=400))

        return ts.normalize() in self._valid_market_days_buffer
    

    def _ts_is_between_time(self, ts: pd.Timestamp) -> bool:
        """
        Check if the given timestamp is within the specified time range.

        This method respects the 'inclusive' parameter to determine how to treat range boundaries.

        Args:
            ts (pd.Timestamp): The timestamp to check.

        Returns:
            bool: True if the timestamp is within the specified time range, False otherwise.

        Raises:
            ValueError: If the inclusive value is invalid.
        """
        if self.between_time is None:
            return True
        
        ts_time = ts.time()

        if self.inclusive == 'neither':
            return self.between_time[0] < ts_time < self.between_time[1]
        elif self.inclusive == 'both':
            return self.between_time[0] <= ts_time <= self.between_time[1]
        elif self.inclusive == 'left':
            return self.between_time[0] <= ts_time < self.between_time[1]
        elif self.inclusive == 'right':
            return self.between_time[0] < ts_time <= self.between_time[1]
        
        raise ValueError(f'Invalid inclusive value: {self.inclusive}')
    

    def _ts_valid(self, ts: pd.Timestamp) -> bool:

        if not self._ts_in_valid_market_days(ts):
            return False
        
        if not self._ts_is_between_time(ts):
            return False
        
        return True


    def _next_ts(self):
        """
        Generate the next valid timestamp.

        Returns:
            pd.Timestamp: The next valid timestamp.
        """
        next_ts = self._current_ts + self.freq

        # Keep incrementing until we find a valid timestamp
        while not self._ts_is_between_time(next_ts) or not self._ts_in_valid_market_days(next_ts):
            next_ts += self.freq

        if self.freq >= pd.Timedelta(days=1):  # for daylight savings 
            next_ts = next_ts.normalize()

        return next_ts
    

    def reset(self):
        """Reset the generator to the starting timestamp."""
        self._current_ts = self.start_ts
        

    def __iter__(self):
        """Make the class iterable."""
        return self
    

    def __next__(self):
        """
        Get the next timestamp in the sequence.

        Returns:
            pd.Timestamp: The next valid timestamp.
        """
        self._current_ts = self._next_ts()
        return self._current_ts
    
    
    @overload
    def __getitem__(self, key: int) -> pd.Timestamp: ...
    @overload
    def __getitem__(self, key: slice) -> list[pd.Timestamp]: ...
    
    def __getitem__(self, key: int | slice) -> pd.Timestamp | list[pd.Timestamp]:
        """
        Allow indexing and slicing of the generator.

        Args:
            key (int | slice): The index or slice to retrieve.

        Returns:
            pd.Timestamp | list[pd.Timestamp]: The timestamp(s) at the specified index/slice.

        Raises:
            ValueError: If negative indices are used or if a slice is missing a stop value.
            TypeError: If the key is neither an integer nor a slice.
        """
        if isinstance(key, int):
            if key < 0:
                raise ValueError('Negative indices are not supported')
            
            # Iterate to the desired index
            result = self._current_ts
            for _ in range(key):
                result = next(self)
            self.reset()
            return result.tz_localize(None) if self.strip_tzinfo else result

        elif isinstance(key, slice):
            result = []

            start, stop, step = key.start or 0, key.stop, key.step or 1
            if stop is None:
                raise ValueError('A "stop" must be provided for slices')
            if start < 0:
                raise ValueError('Negative start indices are not supported')

            # Collect timestamps for the slice
            for i in range(start, stop, step):
                result.append(self[i])

            return result
        
        raise TypeError("Index must be an integer or a slice")
    

def get_market_early_close_ts(date: datetime.date, in_out_tz: str = DATA_TIME_ZONE) -> pd.Timestamp | None:
    early_closes = NYSE_CALENDAR.early_closes(NYSE_CALENDAR.schedule(start_date=date, end_date=date, tz=in_out_tz))
    if early_closes is not None and len(early_closes) == 1:
        early_close_ts = early_closes.iloc[0]['market_close'].tz_localize(None)
        return pd.Timestamp(early_close_ts)
    return None