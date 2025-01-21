import pytest
import pandas as pd
import copy
import pandas.testing as tm

from phantom_core.ohlcv import fill_ohlcv


class TestFillOHLCV:

    def test_1(self, ohlcv_df: pd.DataFrame):
        expected = copy.deepcopy(ohlcv_df)
        actual = fill_ohlcv(ohlcv_df)
        tm.assert_frame_equal(actual, expected)

    def test_2(self, ohlcv_df: pd.DataFrame):
        ohlcv_df.iloc[-1] = None
        filled = fill_ohlcv(ohlcv_df)
        actual = filled.iloc[-1].to_dict()
        expected = {
            'volume': 0.0,
            'vwap': 0.0,
            'transactions': 0.0,
            'close': 150.8,
            'open': 150.8,
            'high': 150.8,
            'low': 150.8,
            'avg_size': 0.0,
            'ticker': 'AAPL'
        }
        assert actual == expected

    def test_3(self, ohlcv_df: pd.DataFrame):
        ohlcv_df.iloc[0] = None
        filled = fill_ohlcv(ohlcv_df)
        actual = filled.iloc[0].to_dict()
        expected = {
            'volume': 0.0,
            'vwap': 0.0,
            'transactions': 0.0,
            'close': 150.45,
            'open': 150.45,
            'high': 150.45,
            'low': 150.45,
            'avg_size': 0.0,
            'ticker': 'AAPL'
        }
        assert actual == expected

    def test_4(self, ohlcv_df: pd.DataFrame):
        ohlcv_df.iloc[2] = None
        filled = fill_ohlcv(ohlcv_df)
        actual = filled.iloc[2].to_dict()
        expected = {
            'volume': 0.0,
            'vwap': 0.0,
            'transactions': 0.0,
            'close': 150.55,
            'open': 150.55,
            'high': 150.55,
            'low': 150.55,
            'avg_size': 0.0,
            'ticker': 'AAPL'
        }
        assert actual == expected
