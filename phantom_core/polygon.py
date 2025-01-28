import pandas as pd
from datetime import datetime, date, time
from typing import Any, Literal
import os

from polygon import RESTClient
from polygon.websocket import WebSocketClient as PolygonWebSocketClient
from polygon.websocket.models.common import Feed, Market
from polygon.rest.models import LastTrade
from polygon.websocket.models import EquityAgg

from .datasource import DataTimeframe, pg_5m_ohlcv_table, Ticker
from .constants import DATA_TIME_ZONE, PERIOD_CNAME
from .ohlcv import clean_ohlcv, OHLCVAgg
from .market_dataframe import MarketDataFrame
from .cache import ttl_cached


def _response_empty(resp: Any) -> bool:
    """
    Check if the response is empty.

    Args:
        resp (Any): The response object to check.

    Returns:
        bool: True if the response is empty, False otherwise.
    """
    return len(resp.__dict__) == 0


def download_ohlcv_data(
    ticker: str, 
    timeframe: DataTimeframe,
    start: pd.Timestamp, 
    end: pd.Timestamp, 
    client: RESTClient | None = None, 
    tz_in: str | None = None,
    tz_out: str = DATA_TIME_ZONE,
    handle_missing_timestamps: bool = False,
    between_time: tuple[time, time] | None = None, 
    between_time_inclusive: Literal['left', 'right', 'both', 'neither'] = 'both',
    respect_valid_market_days: bool = True,
    add_ticker_to_cols: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV (Open, High, Low, Close, Volume) data from Polygon API.

    Supply the timestamps in the desired output timezone. Will handle Polygon timezone converion
    internally.

    This function fetches OHLCV data for a given ticker and time range, processes it,
    and returns a pandas DataFrame with the data.

    Args:
        ticker (str): The stock ticker symbol.
        timeframe (DataTimeframe): The timeframe for the data (e.g., DataTimeframe.MIN_1, DataTimeframe.MIN_5, DataTimeframe.DAILY).
        start (pd.Timestamp): The start date/time for the data range.
        end (pd.Timestamp): The end date/time for the data range.
        client (RESTClient | None): The Polygon API client. If None, a new client will be created.
        tz_in (str | None): The input timezone for start and end if they are not timezone-aware.
        tz_out (str): The output timezone for the returned data. Defaults to DATA_TIME_ZONE.
        handle_missing_timestamps (bool): If True, handle missing timestamps in the data. Defaults to False.
        between_time (tuple[time, time] | None): If provided, filter data to be between these times of day. Defaults to None.
        between_time_inclusive (Literal['left', 'right', 'both', 'neither']): Specifies which endpoints of between_time are inclusive. Defaults to 'both'.
        respect_valid_market_days (bool): If True, only include data from valid market days. Defaults to True.

    Returns:
        pd.DataFrame: The downloaded OHLCV data with columns for open, high, low, close, and volume.
                      The index is a DatetimeIndex with the specified timezone.

    Raises:
        ValueError: If the input timestamps are not timezone-aware and no input timezone is provided.
        ValueError: If the API request fails or returns empty data.

    Note:
        This function handles timezone conversion and ensures that the returned data
        is properly formatted and filtered according to the specified parameters.
    """
    if client is None:
        client = RESTClient(api_key=os.getenv('POLYGON_API_KEY'), trace=False)


    def validate_convert_tz_for_pg(ts: datetime | pd.Timestamp | date) -> pd.Timestamp:
        """
        Validate and convert the input timestamp to a timezone-aware Timestamp in UTC.

        Args:
            ts (datetime | pd.Timestamp | date): The input timestamp to validate.

        Returns:
            pd.Timestamp: A timezone-aware Timestamp in UTC.

        Raises:
            ValueError: If the input timestamp is not timezone-aware and no input timezone is provided.
        """
        ts = pd.Timestamp(ts)

        if ts.tzinfo is None:
            if tz_in is not None:
                ts = ts.tz_localize(tz_in)
            else:
                raise ValueError('Timestamp is not localized and no timezone provided')
        
        return ts.tz_convert('UTC')
    
    
    start_utc = validate_convert_tz_for_pg(start)
    end_utc = validate_convert_tz_for_pg(end)

    timespan, multiplier = timeframe.timespan, timeframe.multiplier

    # Fetch data from Polygon API
    candles = []
    pg_generator = client.list_aggs(ticker=ticker, multiplier=multiplier, timespan=timespan, from_=start_utc, to=end_utc, limit=50_000)
    for response in pg_generator:
        if not _response_empty(response):
            candles.append(response.__dict__)
    ohlcv = pd.DataFrame(candles)

    # normalize the timestamp column name
    ohlcv.rename(columns={'timestamp': PERIOD_CNAME}, inplace=True)
    
    # convert to pd.Timestamp
    ohlcv[PERIOD_CNAME] = pd.to_datetime(ohlcv[PERIOD_CNAME], unit='ms')

    # set the timestamps as the index
    ohlcv.set_index(PERIOD_CNAME, inplace=True)

    # Polygon returns UTC timestamps, so convert to the desired output timezone and strip tzinfo
    assert isinstance(ohlcv.index, pd.DatetimeIndex)
    ohlcv.index = ohlcv.index.tz_localize('UTC').tz_convert(tz_out).tz_localize(None)

    # sort the index
    ohlcv.sort_index(inplace=True)

    # handle between_time if applicable
    if timeframe < DataTimeframe.DAILY and between_time is not None:
        ohlcv = ohlcv.between_time(start_time=between_time[0], end_time=between_time[1], inclusive=between_time_inclusive) # type: ignore
    
    ohlcv.drop(columns=['otc'], inplace=True)

    # data is tzinfo-stripped and in the desired output timezone,
    # so, convert start and end to match
    start_stripped = start_utc.tz_convert(tz_out).tz_localize(None)
    end_stripped = end_utc.tz_convert(tz_out).tz_localize(None)

    if handle_missing_timestamps:
        ohlcv = clean_ohlcv(
            df=ohlcv, 
            timeframe=timeframe,
            start=start_stripped,
            end=end_stripped,
            between_time=between_time,
            between_time_inclusive=between_time_inclusive,
            respect_valid_market_days=respect_valid_market_days
        )

    # ohlcv.index += timeframe

    if add_ticker_to_cols:
        ohlcv['ticker'] = ticker

    return ohlcv


def get_backfill_ohlcv(
        ticker: str, 
        timeframe: DataTimeframe,
        backfill_num_weeks: int, 
        end_ts: pd.Timestamp | None = None,
        extended_trading_hours: bool = False,
    ) -> MarketDataFrame:
    """

    TODO: handle lag; historical API has a lag that isn't accounted for which will be a problem at live
    
    Retrieve historical OHLCV (Open, High, Low, Close, Volume) data for a given ticker.

    Uses US/Eastern timezone.

    This function downloads historical data, handles missing timestamps, and prepares
    the data as a MarketDataFrame suitable for further processing.

    Args:
        ticker (str): The stock ticker symbol.
        timespan (str): The timespan for each candle (e.g., 'minute', 'hour', 'day').
        multiplier (int): The multiplier for the timespan (e.g., 5 for 5-minute candles).
        backfill_num_weeks (int): The number of weeks to look back for historical data.
        regular_session_hours (bool): If True, only include data from regular trading hours.

    Returns:
        MarketDataFrame: A MarketDataFrame containing the historical OHLCV data, with additional
                         'ticker' and 'table' columns. The DataFrame has appropriate column level names
                         and has been processed to handle missing timestamps.

    Raises:
        ValueError: If an unsupported timespan is provided.

    Note:
        - Will always `floor` the provided `end_ts` to the nearest `timeframe`
        - The function uses UTC for initial calculations and then converts to DATA_TIME_ZONE.
        - Missing timestamps are handled using the `handle_missing_timestamps` function.
        - The resulting DataFrame includes data up to the most recent complete candle.
    """

    if end_ts is None:
        _end_ts = pd.Timestamp.now(tz='US/Eastern')

    else:
        if end_ts.tzinfo is None:
            raise ValueError('end_ts must have timezone info')
        _end_ts = end_ts

    _end_ts = _end_ts.floor(timeframe.to_pandas_offset_str()) - timeframe   # subtract timeframe to account for start timestamps in the bar data

    backfill_ts = (_end_ts - pd.Timedelta(weeks=backfill_num_weeks))

    dummy_date = date(1, 1, 1)

    if extended_trading_hours:
        between_time = (
            time(hour=4), 
            (datetime.combine(dummy_date, time(hour=20)) - timeframe).time()
        )
    else:
        between_time = (
            time(hour=9, minute=30), 
            (datetime.combine(dummy_date, time(hour=16)) - timeframe).time()
        )

    ohlcv = download_ohlcv_data(
        ticker=ticker,
        timeframe=timeframe,
        start=backfill_ts,
        end=_end_ts,
        handle_missing_timestamps=True,
        between_time=between_time,
        between_time_inclusive='both',
        respect_valid_market_days=True
    )

    ohlcv['ticker'] = ticker
    ohlcv['table'] = pg_5m_ohlcv_table.db_name  # TODO: make this configurable

    ohlcv = MarketDataFrame(ohlcv)
    ohlcv.set_col_level_names(names=[ohlcv.field_col_level_name])

    return ohlcv


def get_polygon_websocket_client(tickers: list[str], api_key: str | None = None) -> PolygonWebSocketClient:
    """
    Get Polygon WebSocket client. Subscribes to 1-minute aggs for the provided tickers.

    Use by defining a sync handle_message function then call `ws_client.run(handle_message)`.
    """

    if api_key is None:
        api_key = os.environ.get('POLYGON_API_KEY')
        if api_key is None:
            raise ValueError('POLYGON_API_KEY not set')
        
    ws_client = PolygonWebSocketClient(
        api_key=os.environ['POLYGON_API_KEY'],
        feed=Feed.RealTime,
        market=Market.Stocks,
        subscriptions=[f"AM.{ticker}" for ticker in tickers],
        verbose=True
    )
    return ws_client


def get_last_price(ticker: str) -> float:
    pg = RESTClient()
    last_trade = pg.get_last_trade(ticker)
    assert isinstance(last_trade, LastTrade)
    assert last_trade.price is not None
    last_price = float(last_trade.price)
    return last_price


@ttl_cached(20)
def get_last_price_cached(ticker: str) -> float:
    return get_last_price(ticker)


def convert_pg_agg(pg_agg: EquityAgg) -> OHLCVAgg:
    data = pg_agg.__dict__

    ticker = Ticker(data['symbol'].upper())

    if data['event_type'] == 'AM':
        timeframe = DataTimeframe.MIN_1
    else:
        raise ValueError(f"Unsupported event type: {data['event_type']}")
    
    start_ts = pd.to_datetime(data['start_timestamp'], unit='ms')\
        .tz_localize('UTC')\
        .tz_convert('US/Eastern')\
        .tz_localize(None)\
        .to_pydatetime()
    end_ts = pd.to_datetime(data['end_timestamp'], unit='ms')\
        .tz_localize('UTC')\
        .tz_convert('US/Eastern')\
        .tz_localize(None)\
        .to_pydatetime()
    
    volume = data['volume'] / 10   # TODO: needs verification

    return OHLCVAgg.create(
        ticker=ticker,
        timeframe=timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
        open=float(data['open']),
        high=float(data['high']),
        low=float(data['low']),
        close=float(data['close']),
        volume=float(volume),
        vwap=float(data['aggregate_vwap']),
        avg_size=float(data['average_size'])
    )