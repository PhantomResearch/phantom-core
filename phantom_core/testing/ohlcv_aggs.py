import datetime
import pandas as pd

from ..ohlcv import OHLCVAgg, OHLCVAggSpec
from ..datasource import Ticker, DataTimeframe


candle_5m_data = {
    'timeframe': datetime.timedelta(minutes=5),
    'start_ts': pd.Timestamp('2025-01-28 10:00:00').to_pydatetime(),
    'end_ts': pd.Timestamp('2025-01-28 10:05:00').to_pydatetime(),
    'open': 10.0,
    'close': 20.0,
    'high': 25.0,
    'low': 5.0,
    'volume': 100.00,
    'vwap': 25.0,
    'transactions': 10.0
}

def get_5m_1m_aggs() -> tuple[OHLCVAggSpec, OHLCVAgg, list[OHLCVAgg]]:
    
    agg_5m_expected = OHLCVAgg.create(ticker=Ticker('AAPL'), **candle_5m_data)

    min1_aggs = []

    agg = OHLCVAgg.create(
        ticker=Ticker('AAPL'), 
        timeframe=DataTimeframe.MIN_1,
        start_ts=pd.Timestamp('2025-01-28 10:00:00').to_pydatetime(),
        end_ts=pd.Timestamp('2025-01-28 10:01:00').to_pydatetime(),
        open=10.0,
        high=20.0,
        low=10.0,
        close=15.0,
        volume=20.0,
        vwap=1.0,
        transactions=2.0,
    )
    min1_aggs.append(agg)

    agg = OHLCVAgg.create(
        ticker=Ticker('AAPL'), 
        timeframe=DataTimeframe.MIN_1,
        start_ts=pd.Timestamp('2025-01-28 10:01:00').to_pydatetime(),
        end_ts=pd.Timestamp('2025-01-28 10:02:00').to_pydatetime(),
        open=15.0,
        high=20.0,
        low=15.0,
        close=20.0,
        volume=20.0,
        vwap=2.0,
        transactions=2.0,
    )
    min1_aggs.append(agg)

    agg = OHLCVAgg.create(
        ticker=Ticker('AAPL'), 
        timeframe=DataTimeframe.MIN_1,
        start_ts=pd.Timestamp('2025-01-28 10:02:00').to_pydatetime(),
        end_ts=pd.Timestamp('2025-01-28 10:03:00').to_pydatetime(),
        open=20.0,
        high=25.0,
        low=10.0,
        close=10.0,
        volume=20.0,
        vwap=3.0,
        transactions=2.0,
    )
    min1_aggs.append(agg)

    agg = OHLCVAgg.create(
        ticker=Ticker('AAPL'), 
        timeframe=DataTimeframe.MIN_1,
        start_ts=pd.Timestamp('2025-01-28 10:03:00').to_pydatetime(),
        end_ts=pd.Timestamp('2025-01-28 10:04:00').to_pydatetime(),
        open=10.0,
        high=10.0,
        low=5.0,
        close=5.0,
        volume=20.0,
        vwap=4.0,
        transactions=2.0,
    )
    min1_aggs.append(agg)

    agg = OHLCVAgg.create(
        ticker=Ticker('AAPL'), 
        timeframe=DataTimeframe.MIN_1,
        start_ts=pd.Timestamp('2025-01-28 10:04:00').to_pydatetime(),
        end_ts=pd.Timestamp('2025-01-28 10:05:00').to_pydatetime(),
        open=6.0,
        high=20.0,
        low=5.0,
        close=20.0,
        volume=20.0,
        vwap=25.0,
        transactions=2.0,
    )
    min1_aggs.append(agg)

    return agg_5m_expected.spec, agg_5m_expected, min1_aggs