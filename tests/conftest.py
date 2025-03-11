import pytest
import pandas as pd
import copy

@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    index = pd.date_range(start='2024-03-20 09:30:00', periods=5, freq='5min', tz='America/New_York')
    df = pd.DataFrame({
        'volume': [1000, 1500, 800, 2000, 1200],
        'vwap': [150.25, 150.50, 150.40, 150.75, 150.60],
        'transactions': [50, 75, 40, 100, 60],
        'close': [150.30, 150.55, 150.35, 150.80, 150.65],
        'open': [150.20, 150.45, 150.45, 150.70, 150.55],
        'high': [150.35, 150.60, 150.50, 150.85, 150.70],
        'low': [150.15, 150.40, 150.30, 150.65, 150.50],
        'avg_size': [20, 20, 20, 20, 20],
        'ticker': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL']
    }, index=index)
    return copy.deepcopy(df)