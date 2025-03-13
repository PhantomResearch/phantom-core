import datetime
from pickle import NONE
from typing import Literal, Annotated, overload
import numpy as np
from typing_extensions import Self
import pandas as pd
from datetime import time
from pydantic import PlainValidator, WithJsonSchema, BaseModel, model_validator
from sqlalchemy.engine import Engine

from .dataframe_transforms import copy_constant_col_to_all_rows
from .utils import get_first_nonnull_ts
from .dataframe_transforms import reindex_timeseries_df
from .datasource import DataTimeframe, Ticker
# from .market_dataframe import MarketDataFrame


OHLCV_CNAMES = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']


pdTimestamp = Annotated[
    pd.Timestamp,
    PlainValidator(lambda x: pd.Timestamp(x)),
    WithJsonSchema({"type": 'date-time'})
]


def _infer_transactions(volume: float, average_size: float) -> int:
    return int(volume / average_size)


def _infer_average_size(volume: float, transactions: float) -> float:
    if transactions == 0.0:
        return 0.0
    return volume / transactions


def fill_ohlcv(
    df: pd.DataFrame,
    constant_cnames: list[str] = ['ticker', 'table'],
    fill_zero_cnames: list[str] = ['volume', 'vwap', 'transactions', 'avg_size']
) -> pd.DataFrame:
    """
    Fill missing values in OHLCV (Open, High, Low, Close, Volume) data.

    This function assumes the ticker existed throughout the provided datetime range,
    but there are missing timestamps due to no activity.

    It operates on the provided timestamps; does not do any reindexing or validation of timestamps.

    Args:
        df (pd.DataFrame): Input DataFrame containing OHLCV data.
        constant_cnames (list[str], optional): Column names to copy constant values across all rows. 
            Defaults to ['ticker', 'table'].
        fill_zero_cnames (list[str], optional): Column names to fill missing values with 0. 
            Defaults to ['volume', 'vwap', 'transactions'].

    Returns:
        pd.DataFrame: DataFrame with filled OHLCV data.

    Notes:
        - For constant columns (e.g. ticker), copies the single unique non-null value to all rows
        - Fills missing values for volume, vwap, and transactions with 0
        - Forward fills close prices
        - Uses the first non-null open price to fill any missing close prices at the beginning
        - Fills missing open, high, and low prices with the close price
        - Asserts that no null values remain after filling
        - Does not insert missing rows - use `reindex_timeseries_df` first if needed
    """

    for cname in constant_cnames:
        if cname in df.columns:
            df = copy_constant_col_to_all_rows(df, cname)

    for cname in fill_zero_cnames:
        if cname in df.columns:
            df[cname] = df[cname].fillna(0)
    
    df['close'] = df['close'].ffill()

    first_open = df['open'].dropna().iloc[0]
    df['close'] = df['close'].fillna(first_open)

    for cname in ['open', 'high', 'low']:
        df[cname] = df[cname].fillna(df['close'])

    assert df[OHLCV_CNAMES].isnull().sum().sum() == 0

    return df


def clean_ohlcv(
    df: pd.DataFrame, 
    timeframe: DataTimeframe | pd.Timedelta, 
    start: pd.Timestamp | None = None, 
    end: pd.Timestamp | None = None, 
    between_time: tuple[time, time] | None = None, 
    between_time_inclusive: Literal['left', 'right', 'both', 'neither'] = 'both',
    respect_valid_market_days: bool = True,
    bfill_data_start_threshold: pd.Timedelta | Literal['default'] = 'default',
    copy_constant_cols: list[str] = ['ticker', 'table']
) -> pd.DataFrame:
    """
    Handle missing timestamps in OHLCV (Open, High, Low, Close, Volume) data.

    Assumes a constant timezone throughout. This function reindexes the input DataFrame
    to a specified frequency and time range, fills missing values, and handles various
    data integrity issues.

    Notes on time-related arguments:
        - start and end are both optional. you can provide one or both.
        - for start and end, if provided, must have timezone info
        - if start and end are provided, they must have the same timezone
        - between_time, if provided, should be in the same timezone as start and end (user must ensure this!)
        - if start not provided, the function will use the first timestamp in the DataFrame
        - if end not provided, the function will use the last timestamp in the DataFrame
        - if the range of start and/or end is not covered by the data the function will attempt to extrapolate.
        

    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.
        timeframe (DataTimeframe): Desired frequency for reindexing.
        start (pd.Timestamp | None): Start timestamp for reindexing. If None, uses the first timestamp in df.
        end (pd.Timestamp | None): End timestamp for reindexing. If None, uses the last timestamp in df.
        between_time (tuple[time, time] | None): Tuple of (start_time, end_time) to filter timestamps within each day.
        between_time_inclusive (Literal['left', 'right', 'both', 'neither']): How to handle inclusive intervals for between_time filtering.
        respect_valid_market_days (bool): If True, only include valid market days in the reindexed DataFrame.
        bfill_data_start_threshold (pd.Timedelta | Literal['default']): Threshold for backward filling at the start of the data.
        copy_constant_cols (list[str]): Columns to copy to all rows.

    Returns:
        pd.DataFrame: Processed DataFrame with handled missing timestamps and filled values.

    Raises:
        ValueError: If there are issues with ticker or table columns having multiple or no unique values.
        AssertionError: If the input DataFrame's index is not a DatetimeIndex or if null values remain after processing.

    Note:
        - See LucidChart
        - Assumes input DataFrame has columns for OHLCV data and optionally 'ticker' and 'table' columns.
        - Fills missing values for volume, vwap, and transactions with 0.
        - Forward fills close prices.
        - Fills missing open, high, and low prices with the close price.
        - Handles cases where the first non-null timestamp is not at the beginning of the DataFrame.
        - If bfill_data_start_threshold is 'default', it sets to 1 day for daily or longer timeframes,
          and 60 minutes for shorter timeframes.
    """
    
    # validate timestamps
    if start is not None and start.tzinfo is None:
        raise ValueError("start must have timezone info")
    if end is not None and end.tzinfo is None:
        raise ValueError("end must have timezone info")
    if start is not None and end is not None and str(start.tzinfo) != str(end.tzinfo):
        raise ValueError("start and end must have the same timezone")
    
    df = reindex_timeseries_df(
        df=df,
        freq=timeframe,
        start=start,
        end=end,
        between_time=between_time,
        between_time_inclusive=between_time_inclusive,
        respect_valid_market_days=respect_valid_market_days 
    )

    if df.isnull().sum().sum() == 0:
        return df

    assert isinstance(df.index, pd.DatetimeIndex)

    first_observed_ts = get_first_nonnull_ts(df, how='any')

    for cname in copy_constant_cols:
        if cname in df.columns:
            df = copy_constant_col_to_all_rows(df, cname)

    if bfill_data_start_threshold == 'default':

        if timeframe >= DataTimeframe.DAILY:
            bfill_data_start_threshold = pd.Timedelta(days=1)

        else:
            bfill_data_start_threshold = pd.Timedelta(minutes=60)

    if first_observed_ts - df.index[0] <= bfill_data_start_threshold:
        return fill_ohlcv(df)

    before_df = df.loc[:first_observed_ts].iloc[:-1].copy()
    after_df = df.loc[first_observed_ts:].copy()

    after_df = fill_ohlcv(after_df)

    df = pd.concat([before_df, after_df], axis=0)

    assert df.loc[first_observed_ts:].isnull().sum().sum() == 0

    return df


class OHLCVAggSpec(BaseModel):
    ticker: Ticker
    timeframe: datetime.timedelta
    offset: datetime.timedelta = datetime.timedelta(0)

    @model_validator(mode='after')
    def validate_ohlcvaggspec(self) -> Self:
        
        if self.offset < pd.Timedelta(0):
            raise ValueError("Offset must be positive")
        
        if self.offset > pd.Timedelta(0):
        
            if self.offset % pd.Timedelta(minutes=1) != 0:
                raise ValueError("Offset must be a multiple of 1 minute")
            
            if self.offset > self.timeframe:
                raise ValueError("Offset must be less than or equal to the timeframe")
            
            if self.offset >= pd.Timedelta(hours=1):
                raise NotImplementedError("Offset must be less than 1 hour")
            
        return self
    
    @classmethod
    def create(
        cls, 
        ticker: Ticker | str, 
        timeframe: datetime.timedelta | pd.Timedelta | DataTimeframe, 
        offset: datetime.timedelta | pd.Timedelta = datetime.timedelta(0)
    ) -> "OHLCVAggSpec":
        if isinstance(ticker, str):
            ticker = Ticker(ticker)
        if type(timeframe) is pd.Timedelta:
            timeframe = timeframe.to_pytimedelta()
        if type(offset) is pd.Timedelta:
            offset = offset.to_pytimedelta()
        return cls(ticker=ticker, timeframe=timeframe, offset=offset)
        
    
    @property
    def _hash_key(self) -> tuple:
        return (self.ticker, self.timeframe, self.offset)
    
    def __hash__(self) -> int:
        return hash(self._hash_key)
    
    def __eq__(self, other: Self) -> bool:
        return hash(self) == hash(other)
    

class HistoricalOHLCVAggSpec(OHLCVAggSpec):
    start_ts: datetime.datetime
    end_ts: datetime.datetime
    between_time: tuple[time, time] | None = None
    between_time_inclusive: Literal['left', 'right', 'both', 'neither'] = 'left'
    respect_valid_market_days: bool = False
    cleaned: bool = True

    @model_validator(mode='after')
    def validate_historicalohlcvaggspec(self) -> Self:
        if not self.cleaned and self.between_time is not None:
            raise ValueError("between_time must be None if cleaned is False")
        
        if self.start_ts.tzinfo is None or self.end_ts.tzinfo is None:
            raise ValueError(
                "start_ts and end_ts must have timezone info; "
                "remember to specify between_time in the same timezone!"
            )
        
        return self


class OHLCVAgg(OHLCVAggSpec):
    start_ts: datetime.datetime
    end_ts: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    transactions: float
    avg_size: float

    @model_validator(mode="after")
    def validate_ohlcvagg(self) -> Self:

        if self.start_ts.tzinfo is None or self.end_ts.tzinfo is None:
            raise ValueError(
                "start_ts and end_ts must have timezone info; "
            )
        if str(self.start_ts.tzinfo) != str(self.end_ts.tzinfo):
            raise ValueError("start_ts and end_ts must have the same timezone")
        
        if (self.end_ts - self.start_ts) != self.timeframe:
            raise ValueError("end_ts - start_ts must be equal to timeframe")
        
        if self.volume == 0:
            if self.transactions != 0:
                raise ValueError("Volume is 0 but transactions are not 0")
            if self.avg_size != 0:
                raise ValueError("Volume is 0 but avg_size is not 0")
        else:
            if not np.isclose(self.transactions / self.volume, self.avg_size, rtol=0.01):
                raise ValueError("Transactions per volume must be close to average size")

        return self
    
    def to_series(self) -> pd.Series:
        data = self.model_dump(exclude={'ticker', 'timeframe', 'offset'})
        data['ticker'] = str(self.ticker)
        return pd.Series(data=data, name=self.start_ts)
    
    @property
    def spec(self) -> OHLCVAggSpec:
        return OHLCVAggSpec.model_validate(self.model_dump())
    
    @classmethod
    def create(
        cls,
        ticker: Ticker,
        timeframe: DataTimeframe | pd.Timedelta | datetime.timedelta,
        start_ts: datetime.datetime | pd.Timestamp,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        vwap: float,
        end_ts: datetime.datetime | pd.Timestamp | None = None,
        transactions: float | None = None,
        avg_size: float | None = None,
        offset: datetime.timedelta = datetime.timedelta(0)
    ) -> "OHLCVAgg":
        
        if isinstance(start_ts, pd.Timestamp):
            start_ts = start_ts.to_pydatetime()

        if end_ts is not None:
            if isinstance(end_ts, pd.Timestamp):
                end_ts = end_ts.to_pydatetime()
        else:
            end_ts = start_ts + timeframe

        if type(timeframe) != datetime.timedelta:
            timeframe = pd.Timedelta(timeframe).to_pytimedelta()

        if transactions is None:
            if avg_size is None:
                raise ValueError("avg_size must be provided if transactions is not provided")
            transactions = volume * avg_size
        elif avg_size is None:
            avg_size = transactions / volume

        return cls(ticker=ticker, timeframe=timeframe, start_ts=start_ts, end_ts=end_ts, 
                   open=open, high=high, low=low, close=close, volume=volume, vwap=vwap, 
                   transactions=transactions, avg_size=avg_size, offset=offset)
    
    @classmethod
    def from_series(cls, series: pd.Series, **addl_fields) -> Self:
        return cls.model_validate({**series.to_dict(), **addl_fields})
    
    @classmethod
    def create_from_aggs(cls, spec: OHLCVAggSpec, aggs: list["OHLCVAgg"]) -> "OHLCVAgg":

        ticker = spec.ticker
        timeframe = spec.timeframe
        offset = spec.offset

        df = pd.DataFrame([agg.to_series() for agg in aggs]).sort_index()

        start_ts = df['start_ts'].min()
        end_ts = df['end_ts'].max()

        open = float(df.iloc[0]['open'])
        high = float(df['high'].max())
        low = float(df['low'].min())
        close = float(df.iloc[-1]['close'])
        volume = float(df['volume'].sum())
        vwap = float(df['vwap'].iloc[-1])
        transactions = float(df['transactions'].sum())

        return cls.create(
            ticker=ticker, 
            timeframe=timeframe, 
            offset=offset,
            start_ts=start_ts, 
            end_ts=end_ts, 
            open=open, 
            high=high, 
            low=low, 
            close=close, 
            volume=volume, 
            vwap=vwap, 
            transactions=transactions, 
        )
    
    @property
    def _hash_key(self) -> tuple:
        return super()._hash_key + (self.start_ts,)
