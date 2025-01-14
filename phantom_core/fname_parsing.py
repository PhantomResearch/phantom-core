from collections import defaultdict

from .datasource import get_source_tables, SourceTable, DataTimeframe
from .constants import COL_LEVEL_SEP, DEFAULT_COLUMN_LEVEL_NAN


SOURCE_TABLES = get_source_tables()
SOURCE_TABLE_NAME_NAMES = [x.db_name for x in SOURCE_TABLES]
SOURCE_TABLE_NAME_TO_OBJ = {x.db_name: x for x in SOURCE_TABLES}


def group_features_by_source_table(fnames: list[str]) -> dict[SourceTable, list[str]]:
    """
    Group feature names by their source table.

    Args:
        fnames (list[str]): List of feature names.

    Returns:
        dict[SourceTable, list[str]]: Dictionary mapping SourceTable objects to lists of feature names.
    """
    results = defaultdict(list)

    for fname in fnames:
        source_table_name = get_source_table_name_from_fname(fname)
        results[SOURCE_TABLE_NAME_TO_OBJ[source_table_name]].append(fname)

    return results


def group_features_by_timeframe(fnames: list[str]) -> dict[DataTimeframe, list[str]]:
    """
    Group feature names by their associated timeframe.

    Args:
        fnames (list[str]): List of feature names.

    Returns:
        dict[DataTimeframe, list[str]]: Dictionary mapping DataTimeframe objects to lists of feature names.
    """
    results = defaultdict(list)

    source_table_to_fnames = group_features_by_source_table(fnames)

    for source_table, fnames in source_table_to_fnames.items():
        results[source_table.timeframe].extend(fnames)

    return results


def get_fname_components(fname: str) -> tuple[str | None, str | None, str, int | None]:
    """
    Parse a feature name into its components.

    Args:
        fname (str): Feature name to parse.

    Returns:
        tuple[str, str, str, int | None]: Tuple containing (ticker, table, field, lag).
            lag is None if not present in the feature name.

    Raises:
        ValueError: If the feature name format is invalid.
    """

    splits = fname.split(COL_LEVEL_SEP)

    if len(splits) == 4:
        ticker, table, field, lag = splits
    elif len(splits) == 3:
        ticker, table, field = splits
        lag = None
    else:
        raise ValueError(f'invalid fname: {fname}, did not recognize format')
    
    if len(ticker) > 4:
        raise ValueError(f'invalid fname (ticker): {fname}')
    
    if lag is not None:
        if 'lag' not in lag:
            raise ValueError(f'invalid fname (lag): {fname}')
        lag = int(lag.replace('lag', ''))
    
    if table not in SOURCE_TABLE_NAME_NAMES:
        raise ValueError(f'invalid fname (table): {fname}')
    
    if ticker == DEFAULT_COLUMN_LEVEL_NAN:
        ticker = None

    return ticker, table, field, lag
    

def get_source_table_name_from_fname(fname: str) -> str:
    """
    Extract the source table name from a feature name.

    Args:
        fname (str): Feature name to parse.

    Returns:
        str: Name of the source table.

    Raises:
        ValueError: If the extracted table name is invalid.
    """
    _, table, _, _ = get_fname_components(fname)
    
    if table is not None and table in SOURCE_TABLE_NAME_NAMES:
        return table
    else:
        raise ValueError(f'invalid table name: {table}')


def get_fnames_for_datatimeframe(fnames: list[str], timeframe: DataTimeframe) -> list[str]:
    """
    Get feature names for a given DataTimeframe.

    This function filters the input feature names based on the specified DataTimeframe.
    It uses the group_features_by_timeframe function to organize features by timeframe,
    then returns the subset of features corresponding to the given timeframe.

    Args:
        fnames (list[str]): List of feature names to filter.
        timeframe (DataTimeframe): DataTimeframe to filter by.

    Returns:
        list[str]: List of feature names that correspond to the given DataTimeframe.

    Note:
        This function assumes that the group_features_by_timeframe function is defined
        and returns a dictionary mapping DataTimeframe to lists of feature names.
    """

    timeframe_to_fnames = group_features_by_timeframe(fnames)

    return timeframe_to_fnames[timeframe]


def split_fname_into_cname_and_lag(fname: str) -> tuple[str, int]:
    """
    Split a feature name into its column name (cname) and lag components.

    This function parses a feature name and separates it into two parts:
    the column name (cname) and the lag value. The cname is constructed
    by joining the ticker, table, and field components with COL_LEVEL_SEP.

    Args:
        fname (str): Feature name to parse.

    Returns:
        tuple[str, int]: A tuple containing:
            - str: The column name (cname) component of the feature.
            - int: The lag value associated with the feature.

    Raises:
        ValueError: If the input feature name does not have a lag component.

    Example:
        >>> split_fname_into_cname_and_lag("AAPL_||_5m_ohlcv_||_close_||_lag1")
        ('AAPL_||_5m_ohlcv_||_close', 1)
    """

    ticker, table, field, lag  = get_fname_components(fname)

    if lag is None:
        raise ValueError(f'fname does not have a lag component: {fname}')
    

    ticker = ticker or DEFAULT_COLUMN_LEVEL_NAN
    table = table or DEFAULT_COLUMN_LEVEL_NAN

    cname_components = [ticker, table, field]

    return COL_LEVEL_SEP.join(cname_components), lag
