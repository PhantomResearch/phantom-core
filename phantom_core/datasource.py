from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
from typing import Any
import pandas as pd
from pandas.tseries.frequencies import to_offset, DateOffset
from pydantic_core import core_schema

from .constants import DEFAULT_DATA_BASE_PATH


class DataTimeframe(pd.Timedelta, Enum):
    DAILY = 24 * 60 * 60 * 10**9  # 1 day in nanoseconds
    MIN_5 = 5 * 60 * 10**9        # 5 minutes in nanoseconds
    MIN_1 = 60 * 10**9            # 1 minute in nanoseconds
    SECOND = 1 * 10**9            # 1 second in nanoseconds
    TIME = 0                      # 0 for time-based features

    def to_pandas_offset_str(self) -> str:
        offset = to_offset(self)
        assert isinstance(offset, DateOffset)
        return offset.freqstr
    
    @property
    def _timespan_multiplier(self) -> tuple[str, int]:
        if self == DataTimeframe.DAILY:
            return 'day', 1
        elif self == DataTimeframe.MIN_5:
            return 'minute', 5
        elif self == DataTimeframe.MIN_1:
            return 'minute', 1
        elif self == DataTimeframe.SECOND:
            return 'second', 1
        else:
            raise ValueError("Invalid DataTimeframe")
        
    @property
    def timespan(self) -> str:
        return self._timespan_multiplier[0]
    
    @property
    def multiplier(self) -> int:
        return self._timespan_multiplier[1]
    

class SourceTable:
    """
    Represents a table of financial data from a specific data source.

    This class manages the storage location and naming conventions for a particular
    data table, associating it with a data source and timeframe.

    Attributes:
        _dirs (list[str]): List of directory components for the table's path.
        _datasource (DataSource): The data source this table belongs to.
        ingestable (bool): Whether this table can be ingested.
        timeframe (DataTimeframe): The time resolution of the data in this table.
    """

    def __init__(
            self, 
            datasource: "DataSource", 
            table_name: str | list[str], 
            timeframe: DataTimeframe,
            allowable_data_lag: pd.Timedelta = pd.Timedelta(0),
        ):
        """
        Initialize a SourceTable instance.

        Args:
            datasource (DataSource): The data source this table belongs to.
            table_name (str | list[str]): Name or path components of the table.
            timeframe (DataTimeframe): The time resolution of the data in this table.
        """
        # Convert single string table names to a list for consistency
        if isinstance(table_name, str):
            table_name = [table_name]

        self._dirs: list[str] = table_name
        self._datasource = datasource
        self.ingestable = datasource.ingestable
        self.timeframe = timeframe
        self.allowable_data_lag = allowable_data_lag
        # Register this table with its data source
        self._datasource.add_table(self)

    @property
    def path(self, create_if_not_exists: bool = True) -> Path:
        """
        Get the full path to the table's storage location.

        Args:
            create_if_not_exists (bool): If True, create the directory if it doesn't exist.

        Returns:
            Path: The full path to the table's storage location.
        """
        # Start with the base directory of the data source
        p = self._datasource._data_base_dir / self._datasource.data_source_dirname
        
        # Add each component of the table name to the path
        for d in self._dirs:
            p /= d

        # Create the directory if it doesn't exist (if specified)
        if create_if_not_exists:
            p.mkdir(exist_ok=True, parents=True)

        return p

    @property
    def name(self) -> str:
        """
        Get the full name of the table, including the data source name.

        Returns:
            str: The full name of the table.
        """
        return '__'.join([self._datasource.name] + self._dirs) 

    def __str__(self) -> str:
        """
        String representation of the SourceTable.

        Returns:
            str: The name of the table.
        """
        return self.name

    def __repr__(self) -> str:
        """
        Formal string representation of the SourceTable.

        Returns:
            str: The name of the table.
        """
        return str(self)

    @property
    def db_name(self) -> str:
        """
        Get the name of the table as it should appear in the database.

        This method replaces hyphens with underscores to ensure compatibility
        with database naming conventions.

        Returns:
            str: The database-friendly name of the table.
        """
        return self.name.replace('-', '_').lower()

    @property
    def _hash_key(self) -> tuple:
        """
        Get a unique hash key for this SourceTable.

        Returns:
            tuple: A tuple containing the data source, directory components, timeframe, and ingestable status.
        """
        return (self._datasource, tuple(self._dirs), self.timeframe, self.ingestable)

    def __eq__(self, other: object) -> bool:
        """
        Check if this SourceTable is equal to another object.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(other, SourceTable):
            return False
        return self._hash_key == other._hash_key

    def __hash__(self) -> int:
        """
        Get the hash value of this SourceTable.

        Returns:
            int: The hash value.
        """
        return hash(self._hash_key)


class DataSource(ABC):
    """
    Abstract base class representing a source of financial data.

    This class manages a collection of SourceTables and provides a common
    interface for different types of data sources.

    Attributes:
        _data_base_dir (Path): The base directory for all data sources.
        _tables (set[SourceTable]): Set of SourceTables associated with this DataSource.
        ingestable (bool): Whether data from this source can be ingested.
    """

    _data_base_dir: Path = DEFAULT_DATA_BASE_PATH

    def __init__(self, ingestable: bool = True):
        """
        Initialize a DataSource instance.

        Args:
            ingestable (bool): Whether data from this source can be ingested. Defaults to True.
        """
        self._tables: set[SourceTable] = set()
        self.ingestable = ingestable

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Abstract property that should return the name of the data source.

        Returns:
            str: The name of the data source.
        """
        ...

    def add_table(self, table: SourceTable) -> None:
        """
        Add a SourceTable to this DataSource.

        Args:
            table (SourceTable): The table to add.
        """
        if table not in self._tables:
            self._tables.add(table)

    @property
    def data_source_dirname(self) -> str:
        """
        Get the directory name for this data source.

        Returns:
            str: The name of the data source (used as the directory name).
        """
        return self.name

    @property
    def tables(self) -> list[SourceTable]:
        """
        Get a list of all SourceTables associated with this DataSource.

        Returns:
            list[SourceTable]: A list of all tables in this data source.
        """
        return list(self._tables)
    
    @property
    def _hash_key(self) -> str:
        """
        Get a unique hash key for this DataSource.

        Returns:
            str: The name of the data source.
        """
        return self.name

    def __eq__(self, other: object) -> bool:
        """
        Check if this DataSource is equal to another object.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(other, DataSource):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        """
        Get the hash value of this DataSource.

        Returns:
            int: The hash value.
        """
        return hash(self.name)
    

class AlphaVantageDataSource(DataSource):
    """DataSource for Alpha Vantage."""
    name: str = 'alphavantage'

class YFinanceDataSource(DataSource):
    """DataSource for Yahoo Finance."""
    name: str = 'yfinance'

class CXDataSource(DataSource):
    """DataSource for Chart Exchange."""
    name: str = 'chartexchange'

class NYSEDataSource(DataSource):
    """DataSource for New York Stock Exchange."""
    name: str = 'nyse'

class PolygonDataSource(DataSource):
    """DataSource for Polygon."""
    name: str = 'polygon'

class TimeDataSource(DataSource):
    """DataSource for time-related data."""
    name: str = 'time'

# Create instances of each DataSource
# YF_DATASOURCE = YFinanceDataSource()  yfinance is deprecated
CX_DATASOURCE = CXDataSource()
NYSE_DATASOURCE = NYSEDataSource()
AV_DATASOURCE = AlphaVantageDataSource()
PG_DATASOURCE = PolygonDataSource()
TIME_DATASOURCE = TimeDataSource(ingestable=False)

# Define SourceTables for various data types from different sources
alphavantage_historical_options_table = SourceTable(
    datasource=AV_DATASOURCE,
    table_name='HISTORICAL_OPTIONS',
    timeframe=DataTimeframe.DAILY
)

cx_exchange_volume_table = SourceTable(
    datasource=CX_DATASOURCE,
    table_name='exchange-volume',
    timeframe=DataTimeframe.DAILY
)

cx_borrow_fee_table = SourceTable(
    datasource=CX_DATASOURCE,
    table_name='borrow-fee',
    timeframe=DataTimeframe.DAILY
)

cx_failure_to_deliver_table = SourceTable(
    datasource=CX_DATASOURCE,
    table_name='failure-to-deliver',
    timeframe=DataTimeframe.DAILY,
    allowable_data_lag=pd.Timedelta(days=31)
)

cx_short_volume_table = SourceTable(
    datasource=CX_DATASOURCE,
    table_name='short-volume',
    timeframe=DataTimeframe.DAILY
)

pg_5m_ohlcv_table = SourceTable(
    datasource=PG_DATASOURCE,
    table_name='5m_ohlcv',
    timeframe=DataTimeframe.MIN_5,
    allowable_data_lag=pd.Timedelta(minutes=5)
)

pg_1m_ohlcv_table = SourceTable(
    datasource=PG_DATASOURCE,
    table_name='1m_ohlcv',
    timeframe=DataTimeframe.MIN_1,
    allowable_data_lag=pd.Timedelta(minutes=5)
)

pg_daily_ohlcv_table = SourceTable(
    datasource=PG_DATASOURCE,
    table_name='daily_ohlcv',
    timeframe=DataTimeframe.DAILY
)

time_data_table = SourceTable(
    datasource=TIME_DATASOURCE,
    table_name='time_data',
    timeframe=DataTimeframe.TIME
)

# List of all supported data sources
SUPPORTED_DATASOURCES: list[DataSource] = [
    PG_DATASOURCE,
    CX_DATASOURCE,
    NYSE_DATASOURCE,
    AV_DATASOURCE,
    TIME_DATASOURCE,
]

def get_source_tables(timeframe: DataTimeframe | None = None) -> list[SourceTable]:
    """
    Retrieve a list of all SourceTables, optionally filtered by timeframe.

    Args:
        timeframe (DataTimeframe | None): If provided, only return tables with this timeframe.

    Returns:
        list[SourceTable]: A list of SourceTables matching the criteria.
    """
    tables = []
    # Iterate through all supported data sources
    for datasource in SUPPORTED_DATASOURCES:
        # For each data source, iterate through its tables
        for table in datasource.tables:
            # If a timeframe is specified, only include tables with that timeframe
            if timeframe and table.timeframe != timeframe:
                continue
            tables.append(table)
    return tables


class Ticker(str):
    """
    A custom string class representing a stock ticker symbol.

    This class ensures that ticker symbols are always stored in uppercase.
    """

    def __new__(cls, content: str):
        return str.__new__(cls, content.upper())

    def __repr__(self):
        return f"Ticker('{self}')"

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: Any) -> core_schema.CoreSchema:
        """
        Defines the Pydantic core schema for the Ticker class.

        This method is used by Pydantic for validation and serialization.

        Args:
            _source_type: The source type (unused in this implementation).
            _handler: The schema handler (unused in this implementation).

        Returns:
            A CoreSchema object defining the validation and serialization behavior.
        """
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )
