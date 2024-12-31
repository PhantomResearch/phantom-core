from pathlib import Path
import os
import pandas_market_calendars as mcal
from tzlocal import get_localzone


# what date to start the ingest pipeline at
DEFAULT_INGEST_START_DATE = '2014-01-06'


# which chartexchange endpoints the system supports
SUPPORTED_CHARTEXCHANGE_ENDPOINTS = [
    # 'exchange',
    'exchange-volume',
    'borrow-fee',
    # 'chain-summary',
    'failure-to-deliver',
    'reddit-mentions',
    'short-volume',
]


# which alphavantage endpoints/tables the system supports
SUPPORTED_ALPHAVANTAGE_ENDPOINTS = [
    'HISTORICAL_OPTIONS',
]


# top-level directory (Path obj) for raw disk data
PHANTOM_DATA_DIR = os.getenv('PHANTOM_DATA_DIR')
DEFAULT_DATA_BASE_PATH = Path(PHANTOM_DATA_DIR) if PHANTOM_DATA_DIR else Path()


# datetime-esque standards to use globally
DEFAULT_DATE_FORMAT_STR = "%Y-%m-%d"
DEFAULT_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
PERIOD_CNAME = 'timestamp'
PERIOD_FORMAT = DEFAULT_TIMESTAMP_FORMAT
DATA_TIME_ZONE = 'US/Eastern'
LOGGING_TIME_ZONE = 'US/Eastern' if 'PHANTOM_PROD_MODE' in os.environ else str(get_localzone())



# what value to use in a tuple element when there is no value at that index level
DEFAULT_COLUMN_LEVEL_NAN = ''
COL_LEVEL_SEP = '_||_'
# TIME_DATA_TABLE_NAME = 'time_data'


# database constants
STONKS_DATABASE_URL = os.getenv('STONKS_DATABASE_URL')


# The market calendar
NYSE_CALENDAR = mcal.get_calendar('NYSE')


# known ticker aliases
TICKER_ALIASES = {
    ('BRK.A', 'BRK-A', 'BRKA'): 'BRKA',
    ('EXPR', 'EXPRQ') : 'EXPR',
}
def get_ticker_aliases(ticker: str) -> tuple[list[str], str] | None:
    """
    Get ticker aliases and the internal ticker to use for a given ticker.

    If the provided ticker is not listed in `TICKER_ALIASES` will return `None`

    Args:
        ticker (str): _description_

    Returns:
        tuple[list[str], str] | None: If aliases (ticker_aliases: list[str], internal_ticker: str), else None
    """

    # loop through known ticker aliases and return the info if the provided ticker is found
    for ticker_aliases, resolved_ticker in TICKER_ALIASES.items():
        if ticker in ticker_aliases:
            return list(ticker_aliases), resolved_ticker


DEFAULT_TICKERS = [
    'VIXY',
    'GME',
    'XRT',
    'CHWY',
    'KOSS',
    'M',
    'KMX',
    'IWB',
    'STXK',
    'ESPO',
    'SPY',
    'IWM',
    'RATE',
    'VTI',
    'VXF',
    'VBR'
]