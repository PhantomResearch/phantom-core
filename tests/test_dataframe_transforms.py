import pytest
import pandas as pd
import copy
import pandas.testing as tm
import datetime

from phantom_core.dataframe_transforms import reindex_timeseries_df


class TestReindexTimeseriesDF:

    def test_1(self, ohlcv_df: pd.DataFrame):
        """
        Drop a row and assert row was added back
        """
        orig_len = len(ohlcv_df)
        df = ohlcv_df.drop(ohlcv_df.index[2])
        assert len(df) == orig_len - 1
        reindexed = reindex_timeseries_df(df, freq=datetime.timedelta(minutes=5))
        assert len(reindexed) == orig_len

    def test_2(self, ohlcv_df: pd.DataFrame):
        """
        Don't drop a row and assert nothing happened
        """
        expected = copy.deepcopy(ohlcv_df)
        actual = reindex_timeseries_df(ohlcv_df, freq=datetime.timedelta(minutes=5))
        tm.assert_frame_equal(expected, actual)

    def test_3(self, ohlcv_df: pd.DataFrame):
        """
        Reindex should add a row at the end
        """
        original_len = len(ohlcv_df)
        actual = reindex_timeseries_df(
            ohlcv_df, 
            freq=datetime.timedelta(minutes=5),
            end=pd.Timestamp('2024-03-20 09:55', tz='America/New_York')
        )
        assert len(actual) == original_len + 1

    def test_4(self, ohlcv_df: pd.DataFrame):
        """
        Reindex should not add a row
        """
        original_len = len(ohlcv_df)
        actual = reindex_timeseries_df(
            ohlcv_df, 
            freq=datetime.timedelta(minutes=5),
            end=pd.Timestamp('2024-03-20 09:55', tz='America/New_York'),
            start_end_inclusive='left'
        )
        assert len(actual) == original_len

    def test_5(self, ohlcv_df: pd.DataFrame):
        """
        Reindex should add a row at the start
        """
        original_len = len(ohlcv_df)
        actual = reindex_timeseries_df(
            ohlcv_df, 
            freq=datetime.timedelta(minutes=5),
            start=pd.Timestamp('2024-03-20 09:25', tz='America/New_York')
        )
        assert len(actual) == original_len + 1
