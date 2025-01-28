import datetime
from functools import partial
from typing import Literal, Annotated, overload
import numpy as np
from typing_extensions import Self
import pandas as pd
from datetime import time
from pydantic import PlainValidator, WithJsonSchema, BaseModel, model_validator

from polygon.websocket.models import EquityAgg, EventType

from .dataframe_transforms import copy_constant_col_to_all_rows
from .utils import get_first_nonnull_ts, is_list_of_type
from .dataframe_transforms import reindex_timeseries_df, add_null_row_for_timestamp
from .datasource import DataTimeframe, Ticker
from .market_dataframe import MarketDataFrame
from .logging import LoggingMixin, get_logger
from .constants import DATA_TIME_ZONE
from .tqdm import tqdm


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
) -> MarketDataFrame:
    """
    Handle missing timestamps in OHLCV (Open, High, Low, Close, Volume) data.

    Assumes a constant timezone throughout. This function reindexes the input DataFrame
    to a specified frequency and time range, fills missing values, and handles various
    data integrity issues.

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
        MarketDataFrame: Processed DataFrame with handled missing timestamps and filled values.

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
        return MarketDataFrame(df)

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
        return MarketDataFrame(fill_ohlcv(df))

    before_df = df.loc[:first_observed_ts].iloc[:-1].copy()
    after_df = df.loc[first_observed_ts:].copy()

    after_df = fill_ohlcv(after_df)

    df = MarketDataFrame(pd.concat([before_df, after_df], axis=0))

    assert df.loc[first_observed_ts:].isnull().sum().sum() == 0

    return df


class OHLCVAggSpec(BaseModel):
    ticker: Ticker
    timeframe: datetime.timedelta
    offset: datetime.timedelta = datetime.timedelta(0)

    @model_validator(mode='after')
    def validate_ohlcvaggspec(self) -> Self:

        # only supporting MIN_1 and MIN_5 for now
        if self.timeframe not in [DataTimeframe.MIN_1, DataTimeframe.MIN_5]:
            raise NotImplementedError("Only MIN_1 and MIN_5 are supported")
        
        if self.offset < pd.Timedelta(0):
            raise ValueError("Offset must be positive")
        
        if self.offset > pd.Timedelta(0):
        
            if self.offset % pd.Timedelta(minutes=1) != 0:
                raise ValueError("Offset must be a multiple of 1 minute")
            
            if self.offset > self.timeframe:
                raise ValueError("Offset must be less than the timeframe")
            
            if self.offset > pd.Timedelta(hours=1):
                raise NotImplementedError("Offset must be less than 1 hour")
            
        return self
    
    def __hash__(self) -> int:
        return hash((self.ticker, self.timeframe))
    
    def __eq__(self, other: Self) -> bool:
        return hash(self) == hash(other)
    

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
                   transactions=transactions, avg_size=avg_size)
    
    @classmethod
    def from_series(cls, series: pd.Series, **addl_fields) -> Self:
        return cls.model_validate({**series.to_dict(), **addl_fields})
    
    @classmethod
    def create_from_aggs(cls, spec: OHLCVAggSpec, aggs: list["OHLCVAgg"]) -> "OHLCVAgg":

        ticker = spec.ticker
        timeframe = spec.timeframe
        offset = spec.offset

        df = pd.DataFrame([agg.to_series() for agg in aggs]).sort_index()

        start_ts, end_ts = df.index[0], df.index[-1]

        open = float(df.loc[start_ts]['open'])
        high = float(df['high'].max())
        low = float(df['low'].min())
        close = float(df.loc[end_ts]['close'])
        volume = float(df['volume'].sum())
        vwap = float(df['vwap'].sum())                       # TODO: this is wrong
        transactions = float(df['transactions'].sum())
        avg_size = float(df['avg_size'].mean())                     # TODO: check this

        return cls(
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
            avg_size=avg_size
        )
    
    def __hash__(self) -> int:
        return hash((self.ticker, self.timeframe, self.start_ts))