from sqlalchemy.engine import Engine
import pandas as pd
from datetime import timedelta

from ..table_models.ohlcv import OHLCV1M, OHLCV5M
from ...ohlcv import HistoricalOHLCVAggSpec, clean_ohlcv
from .connection import get_postgres_engine
from ..config import DatabaseName
from ...datasource import Ticker
from ...constants import DATA_TIME_ZONE

# Re-export
__all__ = ['HistoricalOHLCVAggSpec', 'get_historical_ohlcv_df_from_db', 'Ticker']

def get_historical_ohlcv_df_from_db(spec: HistoricalOHLCVAggSpec, engine: Engine | None = None) -> pd.DataFrame:
    """
    Retrieve historical OHLCV data from the database based on the provided specification.
    
    Args:
        spec: Specification for the historical OHLCV data to retrieve
        engine: SQLAlchemy engine to use for database connection (optional)
        
    Returns:
        DataFrame containing the historical OHLCV data
        
    Raises:
        ValueError: If the timeframe is not supported
    """
    engine = engine or get_postgres_engine(database=DatabaseName.RAW_INGEST)

    # determine table_name
    if spec.timeframe == timedelta(minutes=1):
        table_name = OHLCV1M.__tablename__
    elif spec.timeframe == timedelta(minutes=5):
        table_name = OHLCV5M.__tablename__
    else:
        raise ValueError(f"Timeframe {spec.timeframe} not supported [yet] in this function")
    
    # Handle timezone conversion for start_ts and end_ts
    start_ts = pd.Timestamp(spec.start_ts)
    end_ts = pd.Timestamp(spec.end_ts)
    
    # Ensure timestamps have timezone info
    # This should be handled by HistoricalOHLCVAggSpec validator
    # but double checking for confidence
    assert start_ts.tzinfo is not None and end_ts.tzinfo is not None
    assert str(start_ts.tzinfo) == str(end_ts.tzinfo)
    
    # Build the SQL query with UTC timestamps (table in UTC)
    query = f"""
    SELECT * FROM {table_name}
    WHERE symbol = '{spec.ticker}'
    AND timestamp >= '{start_ts.tz_convert('UTC').isoformat()}'
    AND timestamp < '{end_ts.tz_convert('UTC').isoformat()}'
    ORDER BY timestamp
    """
    
    # Execute the query and load into DataFrame
    df = pd.read_sql(query, engine, index_col='timestamp', parse_dates=['timestamp'])
    assert isinstance(df.index, pd.DatetimeIndex)
    df.index = df.index.tz_convert(start_ts.tzinfo)  # convert index to time being used in the request (ie, HistoricalOHLCVAggSpec)
    
    # If DataFrame is empty, return it as is
    if df.empty:
        return df
    
    # Rename columns to match expected format if needed
    column_mapping = {
        'symbol': 'ticker',
    }
    df = df.rename(columns=column_mapping)
    
    # Apply cleaning if requested
    if spec.cleaned:
        df = clean_ohlcv(
            df=df,
            timeframe=pd.Timedelta(spec.timeframe),
            # start=start_ts,  # Use the timezone-aware timestamps
            # end=end_ts,
            between_time=spec.between_time,
            between_time_inclusive=spec.between_time_inclusive,
            respect_valid_market_days=spec.respect_valid_market_days
        )
    
    return df

