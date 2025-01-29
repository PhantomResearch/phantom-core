import pytest
import pandas as pd
import copy
import pandas.testing as tm

from phantom_core.ohlcv import OHLCVAgg, OHLCVAggSpec, fill_ohlcv
from phantom_core.testing.ohlcv_aggs import get_5m_1m_aggs


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


@pytest.fixture
def m5m1_aggs() -> tuple[OHLCVAggSpec, OHLCVAgg, list[OHLCVAgg]]:
    return get_5m_1m_aggs()


class TestOHLCVAgg:

    def test_create_from_aggs(self, m5m1_aggs):
        m5_spec, m5_agg, m1_aggs = m5m1_aggs

        m5_agg_created = OHLCVAgg.create_from_aggs(spec=m5_spec, aggs=m1_aggs)

        assert m5_agg == m5_agg_created

