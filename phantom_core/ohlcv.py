from functools import partial
from typing import Literal, Annotated
from typing_extensions import Self
import pandas as pd
from datetime import time
from pydantic import PlainValidator, WithJsonSchema, BaseModel

from polygon.websocket.models import EquityAgg, EventType

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


class PhantomEquityAgg(BaseModel):

    timeframe: DataTimeframe
    symbol: Ticker
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    transactions: float
    average_size: float
    start_timestamp: pdTimestamp
    end_timestamp: pdTimestamp
    event_type: EventType | None = None
    accumulated_volume: float | None = None
    official_open_price: float | None = None
    aggregate_vwap: float | None = None
    otc: bool | None = None

    # @field_validator('symbol', mode='before')
    # @classmethod
    # def ensure_ticker(cls, v):
    #     if type(v) is str:
    #         return Ticker(v)
    #     return v

    @classmethod
    def from_pg_equity_agg(cls, agg: EquityAgg, timeframe: DataTimeframe) -> Self:
        # convert polygon model to dict
        d = agg.__dict__

        # handle timestamps
        assert d.get('start_timestamp') is not None
        assert d.get('end_timestamp') is not None
        d['start_timestamp'] = pd.to_datetime(d['start_timestamp'], unit='ms').tz_localize('UTC').tz_convert(DATA_TIME_ZONE).tz_localize(None)
        d['end_timestamp'] = pd.to_datetime(d['end_timestamp'], unit='ms').tz_localize('UTC').tz_convert(DATA_TIME_ZONE).tz_localize(None)

        # handle transactions
        d['transactions'] = _infer_transactions(volume=d['volume'], average_size=d['average_size'])

        return cls(**agg.__dict__, timeframe=timeframe)


    def to_pd_series(self) -> pd.Series:

        timestamp = self.start_timestamp

        return pd.Series(
            data={
                'open': self.open,
                'high': self.high,
                'low': self.low,
                'close': self.close,
                'volume': self.volume,
                'vwap': self.vwap,
                'transactions': self.transactions,
                'ticker': str(self.symbol),
            },
            name=timestamp,
        )
    
    
    def attach_to_ohlcv(self, ohlcv: pd.DataFrame) -> MarketDataFrame:
        """
        Attach the PhantomEquityAgg to the OHLCV DataFrame.

        This method creates a new row with the aggregate data and appends it to the existing OHLCV DataFrame.
        It handles timestamp conversion, ensures all required fields are present, and sorts the resulting DataFrame.

        Args:
            ohlcv (pd.DataFrame): The existing OHLCV DataFrame to which the aggregate data will be attached.

        Returns:
            pd.DataFrame: The updated OHLCV DataFrame with the new aggregate data attached and sorted by timestamp.

        Raises:
            AssertionError: If any required fields (end_timestamp, open, close, high, low, volume, vwap, average_size) are None.
        """

        ohlcv = pd.concat([ohlcv, self.to_pd_series().to_frame().T])
        ohlcv.sort_index(inplace=True)
        return MarketDataFrame(ohlcv)
    

    @classmethod
    def from_pd_series(cls, s: pd.Series, timeframe: DataTimeframe) -> Self:
        
        start_timestamp = s.name
        assert isinstance(start_timestamp, pd.Timestamp)
        end_timestamp = start_timestamp + timeframe

        if 'ticker' not in s:
            raise ValueError('ticker not in series')
        else:
            ticker = s['ticker']

        if 'average_size' not in s:
            average_size = _infer_average_size(s['volume'], s['transactions'])
        else:
            average_size = s['average_size']
        
        return cls(
            timeframe=timeframe,
            symbol=Ticker(ticker),
            open=s['open'],
            high=s['high'],
            low=s['low'],
            close=s['close'],
            volume=s['volume'],
            vwap=s['vwap'], 
            transactions=s['transactions'],
            average_size=average_size, 
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
    
    
    @classmethod
    def from_bundle(
        cls, 
        bundle: list[Self], 
        timeframe: DataTimeframe,
        end_ts: pd.Timestamp
    ) -> Self:
        
        # Check if end_ts has timezone information
        if end_ts.tzinfo is not None:
            end_ts = end_ts.tz_convert(DATA_TIME_ZONE).tz_localize(None)


        # Ensure all aggregates in the bundle have the same timeframe
        bundle_timeframe = set(agg.timeframe for agg in bundle)
        if len(bundle_timeframe) != 1:
            raise ValueError(f'bundle has more than one timeframe: {bundle_timeframe}')
        bundle_timeframe = bundle_timeframe.pop()
        
        # Check if the bundle timeframe is smaller than or equal to the desired timeframe
        if bundle_timeframe > timeframe:
            raise ValueError(f'bundle_timeframe {bundle_timeframe} > desired timeframe {timeframe}')
        
        symbols = set(agg.symbol for agg in bundle)
        if len(symbols) > 1:
            raise ValueError(f'bundle has more than one symbol: {symbols}')
        symbol = symbols.pop()
        
        # Convert bundle to DataFrame for easier manipulation
        df = pd.DataFrame([agg.to_pd_series() for agg in bundle]).sort_index()
        
        if df.index.max() >= end_ts or  df.index.min() < end_ts - timeframe:
            raise ValueError(
                f'Invalid bundle end timestamps. '
                f'min: {df.index.min()}, '
                f'max: {df.index.max()} '
                f'for end_ts: {end_ts} '
                f'and timeframe: {timeframe}'
            )
        
        assert isinstance(df.index, pd.DatetimeIndex)

        volume = df['volume'].sum()
        no_volume = volume == 0

        if no_volume and df['transactions'].sum() > 0:
            raise ValueError('data validation issue: no_volume in bundle but transactions > 0')
        
        vwap = 0.0 if no_volume else df['vwap'].mul(df['volume']).sum() / volume
        average_size = 0.0 if no_volume else volume / df['transactions'].sum()
              
        # Calculate aggregate values
        open = df['open'].iloc[0]
        close = df['close'].iloc[-1]
        high = df['high'].max()
        low = df['low'].min()
        transactions = df['transactions'].sum()
        average_size = average_size

        # Create and return the new PhantomEquityAgg instance
        return cls(
            timeframe=timeframe,
            symbol=symbol,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=vwap,
            transactions=transactions,
            average_size=average_size,
            start_timestamp=end_ts - timeframe,
            end_timestamp=end_ts,   
        )
    

def bundle_to_df(bundle: list[PhantomEquityAgg]) -> pd.DataFrame:
    """
    Convert a list of PhantomEquityAgg objects to a pandas DataFrame.

    This function takes a list of PhantomEquityAgg objects and converts them into
    a pandas DataFrame. Each PhantomEquityAgg object is converted to a pandas Series
    using its to_pd_series method, and these Series are then combined into a DataFrame.
    The resulting DataFrame is sorted by index.

    Args:
        bundle (list[PhantomEquityAgg]): A list of PhantomEquityAgg objects to convert.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the PhantomEquityAgg objects,
                      sorted by index.

    Note:
        - The resulting DataFrame will have columns corresponding to the attributes
          of the PhantomEquityAgg objects.
        - The index of the DataFrame will be determined by the index of the Series
          returned by the to_pd_series method of each PhantomEquityAgg object.
    """
    return pd.DataFrame([agg.to_pd_series() for agg in bundle]).sort_index()
    

def ohlcv_df_to_aggs(ohlcv: pd.DataFrame, timeframe: DataTimeframe) -> list[PhantomEquityAgg]:
    """
    Convert an OHLCV DataFrame to a list of PhantomEquityAgg instances.

    This function takes an OHLCV DataFrame and converts each row into a PhantomEquityAgg instance.
    It uses the `from_pd_series` method to create individual aggregates and collects them in a list.

    Args:
        ohlcv (pd.DataFrame): The OHLCV DataFrame to convert.
        timeframe (DataTimeframe): The desired timeframe for the resulting aggregates.

    Returns:
        list[PhantomEquityAgg]: A list of PhantomEquityAgg instances.

    Raises:
        ValueError: If the OHLCV DataFrame is empty or if the timeframe is not supported.

    Note:
        - The function uses a loop to iterate through each row of the OHLCV DataFrame.
        - For each row, it creates a PhantomEquityAgg instance using the `from_pd_series` method.
        - The resulting list of aggregates is returned.
    """
    aggs = []
    for _, row in tqdm(ohlcv.iterrows(), total=len(ohlcv), desc='Converting OHLCV to PhantomEquityAggs'):
        aggs.append(PhantomEquityAgg.from_pd_series(row, timeframe=DataTimeframe.MIN_1))
    return aggs

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


def fill_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in OHLCV (Open, High, Low, Close, Volume) data.

    This function assumes the ticker existed throughout the provided datetime range,
    but there are missing timestamps due to no activity.

    It operates on the provided timestamps; does not do any reindexing or validation of timestamps.

    Args:
        df (pd.DataFrame): Input DataFrame containing OHLCV data.

    Returns:
        pd.DataFrame: DataFrame with filled OHLCV data.

    Notes:
        - See LucidChart
        - Fills missing values for volume, vwap, and transactions with 0.
        - Forward fills close prices.
        - Uses the first non-null open price to fill any missing close prices at the beginning.
        - Fills missing open, high, and low prices with the close price.
        - Asserts that no null values remain after filling.
    """

    df['volume'] = df['volume'].fillna(0)
    df['vwap'] = df['vwap'].fillna(0)
    df['transactions'] = df['transactions'].fillna(0)
    
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

    original_reindexed_shape = df.shape
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


class OhlcvDataBundle(dict[tuple[DataTimeframe, Ticker], MarketDataFrame], LoggingMixin):
    """
    A custom dictionary class for storing and managing OHLCV (Open, High, Low, Close, Volume) data.

    This class extends the built-in dict class and includes logging capabilities.
    It uses tuples of DataTimeframe and Ticker as keys, and MarketDataFrame as values.
    """

    def __init__(self):
        """
        Initializes the OhlcvDataBundle with a logger.
        """
        super().__init__()
        self.logger = get_logger(name='OhlcvDataBundle')
        

    def __setitem__(self, key: tuple[DataTimeframe, Ticker], value: MarketDataFrame):
        """
        Sets an item in the OhlcvDataBundle.

        Args:
            key: A tuple containing a DataTimeframe and a Ticker.
            value: A MarketDataFrame containing OHLCV data.

        Raises:
            KeyError: If the key is not a tuple of (DataTimeframe, Ticker).
            ValueError: If the value is not a MarketDataFrame.
        """
        if not isinstance(key, tuple) or len(key) != 2 or not isinstance(key[0], DataTimeframe) or not isinstance(key[1], Ticker):
            raise KeyError("Key must be a tuple of (DataTimeframe, Ticker)")
        if not isinstance(value, MarketDataFrame):
            raise ValueError("Value must be a MarketDataFrame")
        super().__setitem__(key, value)

    @overload
    def __getitem__(self, key: tuple[DataTimeframe, Ticker]) -> MarketDataFrame: ...
    @overload
    def __getitem__(self, key: tuple[DataTimeframe, str]) -> MarketDataFrame: ...
    @overload
    def __getitem__(self, key: DataTimeframe) -> dict[Ticker, MarketDataFrame]: ...
    @overload
    def __getitem__(self, key: Ticker) -> dict[DataTimeframe, MarketDataFrame]: ...
    @overload
    def __getitem__(self, key: str) -> dict[DataTimeframe, MarketDataFrame]: ...

    def __getitem__(self, key):
        """
        Retrieves an item or items from the OhlcvDataBundle based on the provided key.

        This method supports various key types for flexible data retrieval.

        Args:
            key: Can be a tuple of (DataTimeframe, Ticker), a DataTimeframe, a Ticker, or a string.

        Returns:
            Either a single MarketDataFrame or a dictionary of MarketDataFrames, depending on the key type.

        Raises:
            KeyError: If the key type is invalid.
        """
        if isinstance(key, tuple) and len(key) == 2:
            return super().__getitem__(key)
        elif isinstance(key, DataTimeframe):
            return {ticker: df for (tf, ticker), df in self.items() if tf == key}
        elif isinstance(key, Ticker):
            return {tf: df for (tf, ticker), df in self.items() if ticker == key}
        else:
            raise KeyError(f"Invalid key type: {type(key)}")
        

    @property
    def timeframes_in_bundle(self) -> list[DataTimeframe]:
        """
        Returns a list of unique DataTimeframes present in the bundle.

        Returns:
            A list of DataTimeframe objects.
        """
        self_keys = list(self.keys())
        return list(set(tf for (tf, _) in self_keys))
    

    @property
    def tickers_in_bundle(self) -> list[Ticker]:
        """
        Returns a list of unique Ticker symbols present in the bundle.

        Returns:
            A list of Ticker objects.
        """
        self_keys = list(self.keys())
        return list(set(ticker for (_, ticker) in self_keys))


    def _group_aggs_by_timeframe(self, aggs: list[PhantomEquityAgg]) -> dict[DataTimeframe, list[PhantomEquityAgg]]:
        """
        Groups a list of PhantomEquityAgg objects by their timeframes.

        Args:
            aggs: A list of PhantomEquityAgg objects.

        Returns:
            A dictionary where keys are DataTimeframe objects and values are lists of PhantomEquityAgg objects.
        """
        unique_timeframes = list(set(agg.timeframe for agg in aggs))

        result = {tf: [] for tf in unique_timeframes}

        for agg in aggs:
            result[agg.timeframe].append(agg)

        return result


    def update_from_phantom_aggs(
            self, 
            aggs: PhantomEquityAgg | list[PhantomEquityAgg],
            handle_missing_tickers: bool = True,
            truncate_df_heads: bool = True
        ) -> None:
        """
        Updates the OhlcvDataBundle with new data from PhantomEquityAgg objects.

        This method can handle aggregates from different timeframes and fills in missing ohlcv data.

        Args:
            aggs: A single PhantomEquityAgg or a list of PhantomEquityAgg objects.
            handle_missing_tickers: If True, fills in missing data for tickers not present in the new aggregates.
            truncate_df_heads: If True, removes the oldest row when adding a new row to maintain constant dataframe length.

        Raises:
            ValueError: If PhantomEquityAggs for a given timeframe have different start timestamps.

        Note:
            - Assumes a complete buffer of aggregates is provided (ie, messing timestamps are assumed to be zero-volume aggs).
            - Assumes the dataframes are sorted by time.
            - Requires that all PhantomEquityAggs for each timeframe have the same start timestamp.
        """
        
        if isinstance(aggs, PhantomEquityAgg):
            _aggs = [aggs]
        else:
            _aggs = aggs

        assert is_list_of_type(_aggs, PhantomEquityAgg)

        tf_grouped_aggs = self._group_aggs_by_timeframe(aggs=_aggs)

        for tf, aggs_list in tf_grouped_aggs.items():

            ts_to_add = set(agg.start_timestamp for agg in aggs_list)
            if len(ts_to_add) > 1:
                raise ValueError("All PhantomEquityAggs for a given timeframe must have the same start timestamp")
            ts_to_add = ts_to_add.pop()

            missing_tickers_for_tf = set(self[tf].keys())

            for agg in aggs_list:

                agg_ticker = agg.symbol

                df = self[(tf, agg_ticker)]

                df = agg.attach_to_ohlcv(ohlcv=df)
                df = copy_constant_col_to_all_rows(df=df, cname='table')

                if truncate_df_heads:
                    df = df.iloc[1:]

                self[(tf, agg_ticker)] = MarketDataFrame(df)    

                missing_tickers_for_tf.remove(agg_ticker)

            if len(missing_tickers_for_tf) > 0 and handle_missing_tickers:

                for ticker in list(missing_tickers_for_tf):

                    df = self[(tf, ticker)]

                    df = add_null_row_for_timestamp(df=df, ts=ts_to_add)

                    df = fill_ohlcv(df=df)

                    if truncate_df_heads:
                        df = df.iloc[1:]

                    self[(tf, ticker)] = MarketDataFrame(df)

                    self.log(f'Filled missing data for {ticker} at {ts_to_add}')