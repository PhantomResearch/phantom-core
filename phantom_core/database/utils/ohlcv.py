from sqlalchemy.engine import Engine
import pandas as pd

from ..table_models.ohlcv import OHLCV1M, OHLCV5M
from ...ohlcv import HistoricalOHLCVAggSpec


def get_historical_ohlcv_df_from_db(spec: HistoricalOHLCVAggSpec, engine: Engine | None = None) -> pd.DataFrame:
    pass