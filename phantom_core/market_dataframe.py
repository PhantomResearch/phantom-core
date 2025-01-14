""" 
MarketDataFrame Module

This module provides the `MarketDataFrame` class, a custom pandas DataFrame tailored for market data analysis.
It extends pandas DataFrame with specialized functionality including:

- **Metadata Handling**: Manages date-index mappings and NLP modules.
- **Column Name Manipulation**: Supports both string and tuple forms for column names.
- **Natural Language Processing**: Facilitates fuzzy column selection based on natural language queries.
- **Data Visualization**: Includes utilities to plot data completeness over time.
- **Index Management**: Supports conversion between integer and datetime indices.
- **Utility Functions**: Integrates with utilities for date mapping and column filtering.

**Key Classes and Methods:**

- `DateIdxMappings`: A dataclass to store mappings between dates and indices.
  
- `MarketDataFrameNLP`: Handles natural language processing operations on `MarketDataFrame`, providing methods for fuzzy column selection.

- `MarketDataFrame`: 
    - **Initialization & Metadata**
        - `__init__`: Initializes the MarketDataFrame with optional metadata.
        - `_metadata`: Lists custom metadata attributes.
        - `set_default_metadata`: Sets default metadata attributes if not provided.
    - **Column Manipulation**
        - `to_str_cols` / `to_tuple_cols`: Converts column names between string and tuple forms.
        - `add_column_level_single_value`: Adds a new level to the column index.
        - `create_column_single_name`: Creates a new column with a single name.
        - `_make_cname_str` / `_make_cname_tuple`: Converts column names to string or tuple forms.
    - **Index Handling**
        - `to_int_index` / `to_dt_index`: Converts index types between integer and datetime.
    - **NLP and Filtering**
        - `nlp`: Accesses the NLP module for the DataFrame.
        - `filter_cnames`: Filters column names based on provided criteria.
    - **Data Visualization**
        - `plot_column_data_starts`: Plots the cumulative number of columns with data over time.
    - **Properties**
        - `tickers`, `tables`, `num_levels`, and others provide easy access to various attributes of the DataFrame.
        
This module is designed to facilitate efficient market data analysis by enhancing the capabilities of pandas DataFrame with specialized features tailored for financial data.

"""

from datetime import datetime
from pathlib import Path
from typing_extensions import Self
from typing import overload
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import itertools
import re
from typing_extensions import Self

from .utils import get_first_and_last_valid_index, datetime_ify, filter_cnames
from .constants import COL_LEVEL_SEP


def get_date_idx_mappings(df: pd.DataFrame) -> tuple[dict[int, datetime], dict[datetime, int]] | None:
    """
    Looks for a timestamp/date in the index and column names and generates the mappings accordingly
    """

    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index.to_series()

    elif 'date' in df.columns:
        timestamps = pd.to_datetime(df['date'])

    elif 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])

    else:
        return
    
    idx2date: dict[int, datetime] = dict()
    for i, ts in enumerate(timestamps):
        idx2date[i] = ts.to_pydatetime()

    date2idx = {d:i for i,d in idx2date.items()}
    
    return idx2date, date2idx


@dataclass
class DateIdxMappings:
    """
    A dataclass to store mappings between dates and indices.

    Attributes:
        i2d (dict[int, datetime]): A dictionary mapping indices to dates.
        d2i (dict[datetime, int]): A dictionary mapping dates to indices.
    """
    i2d: dict[int, datetime]
    d2i: dict[datetime, int]


class MarketDataFrameNLP:
    """
    A class to handle natural language processing operations on MarketDataFrame.

    This class provides methods for fuzzy column selection based on natural language queries.
    """

    def __init__(self, df: "MarketDataFrame"):
        """
        Initialize the MarketDataFrameNLP object.

        Args:
            df (MarketDataFrame): The MarketDataFrame object to perform NLP operations on.
        """
        self.df = df

    def _return_column(self, cname: tuple):
        """
        Helper method to handle string versus tuple form in the underlying dataframe,
        so as to return the column with its name in the matching form.

        Args:
            cname (tuple): The column name in tuple form.

        Returns:
            pd.Series: The column data.
        """
        if self.df.is_tuple_cols:
            cname_ = self.df._make_cname_tuple(cname)
        else:
            cname_ = self.df._make_cname_str(cname)
        return self.df[cname_]
    

    @staticmethod
    def _normalize_string(s: str):
        """
        Normalize a string by removing parentheses, replacing underscores with spaces,
        removing extra spaces, and converting to lowercase.

        Args:
            s (str): The input string to normalize.

        Returns:
            str: The normalized string.
        """
        # Step 1: Remove parentheses
        normalized_str = re.sub(r'[()]', '', s)

        # Step 2: Replace underscores with spaces
        normalized_str = normalized_str.replace('_', ' ')

        # Step 3: Remove extra spaces
        normalized_str = re.sub(r'\s+', ' ', normalized_str).strip()

        # Step 4: Lowercase and return
        return normalized_str.lower()


    def __getitem__(self, query: str) -> pd.Series:
        """
        Fuzzy single column selection. Helps when column names are long and complex. 
        
        Requirements for the query in order to get a match:
            1. The query is required to have the ticker in it somewhere (lowercase is fine).
            2. All the tokens of the bottom column name must be present in the query.
               Using spaces instead of underscores, etc is supported and encouraged.

        Multi-stage sequential filtering process:
            1. Filter to column names whose ticker is in the query
            2. Filter to column names whose bottom column tokens are all present in the query
               (will return at this point if only a single column makes it past the filter)
            3. Filter to column names whose tokens superset the tokens of the query 
               (ie, all query tokens are present in candidate)
            4. Filter to the single column name whose token Intersection Over Untion (IoU) 
               is the greatest among the final candidates.

        Args:
            query (str): The natural language query to match against column names.

        Returns:
            pd.Series: The matched column data.

        Raises:
            ValueError: If no match is found for the query.
        """

        # process the query into a set of lower case tokens split by whitespace
        query = query.lower()
        query_tokens = set(query.split())

        # first stage of candidates are cnames where the ticker is in the query
        candidate_cnames = [c for c in self.df._tuple_cnames if c[0].lower() in query_tokens]
        if len(candidate_cnames) == 0:
            raise ValueError(f'query not matched ({query})')

        # second stage of candidates are cnames where the tokens of the bottom column name are all in the query
        tmp = []
        for candidate_cname in candidate_cnames:
            bottom_cname = candidate_cname[-1]
            bottom_cname_tokens = set(self._normalize_string(bottom_cname).split())
            if len(bottom_cname_tokens - query_tokens) == 0:
                tmp.append(candidate_cname)
        candidate_cnames = tmp
        if len(candidate_cnames) == 0:
            raise ValueError(f'query not matched ({query})')
        if len(candidate_cnames) == 1:
            return self._return_column(candidate_cnames[0])
        
        # create a normalized version of the
        sep = self.df._col_sep
        normalized_candidate_cnames = [self._normalize_string(self.df._make_cname_str(x)) for x in candidate_cnames]

        # loop over candidates
        match_results = []
        for candidate_cname, normalized_candidate_cname in zip(candidate_cnames, normalized_candidate_cnames):
            
            # tokenize the candidate
            candidate_tokens = set(normalized_candidate_cname.split())

            # compute how many of the query tokens are present in the candidate
            leftover_tokens = query_tokens - candidate_tokens

            # if all query tokens are present in candidate, then this is a final stage candidate
            if len(leftover_tokens) == 0:

                # compute the IoU score between the query and candidate and append the result
                iou = len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)
                match_results.append((candidate_cname, iou))

        # if any final candidates
        if match_results:

            # pull out the candidate cname with the highest IoU score
            best_cname_match, best_iou = max(match_results, key=lambda x: x[1])
            
            # return the column
            return self._return_column(best_cname_match)
        
        # if no final candidates return none
        else:
            raise ValueError(f'query not matched ({query})')


class MarketDataFrame(pd.DataFrame):
    """
    A custom DataFrame class for market data analysis.

    This class extends pandas DataFrame with additional functionality specific to market data,
    including metadata handling, column name manipulation, and data visualization.

    Attributes:
        _metadata (list): List of custom metadata attributes.
        _date_idx_mappings (DateIdxMappings): Mappings between dates and indices.
        _str_tuple_col_sep (str): Separator used for string representation of tuple column names.
        _nlp_module (MarketDataFrameNLP): NLP module for column selection.
        _col_level_names (list): Names of column levels.
    """

    _metadata = ['_date_idx_mappings', '_str_tuple_col_sep', '_nlp_module', '_col_level_names']

    def __init__(self, *args, **kwargs):
        """
        Initialize the MarketDataFrame.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        # if this is a `MarketDataFrame` instantiating itself after a transform,
        # it will pass its metadata on. capture it here
        # loop over metadata keys (above)
        md_kwargs = dict()
        for md_key in self._metadata:

            # if try popping the metadata value out of the kwargs
            md_kwargs[md_key] = kwargs.pop(md_key, None)

        # initialize the `pd.DataFrame`
        super().__init__(*args, **kwargs)

        # parse the captures metadata
        # `md_kwargs` will have keys for all custom metadata
        # but the values can be `None` in which case they need to be computed and set
        for md_key, md_val in md_kwargs.items():
            if md_val is not None:
                setattr(self, md_key, md_val)
            else:
                self.set_default_metadata(md_key)


    def update_underlying_dataframe(self, df: pd.DataFrame, copy_cnames: bool, copy_index: bool) -> "MarketDataFrame":
        """
        Update the underlying DataFrame while preserving metadata.

        Args:
            df (pd.DataFrame): The new DataFrame.
            copy_cnames (bool): Whether to copy column names from the current DataFrame.
            copy_index (bool): Whether to copy the index from the current DataFrame.

        Returns:
            MarketDataFrame: A new MarketDataFrame with updated data and preserved metadata.
        """
        
        md_kwargs = dict()
        for k in self._metadata:
            md_kwargs[k] = getattr(self, k)

        mdf = MarketDataFrame(df, **md_kwargs)

        if copy_cnames:
            mdf.columns = self.columns

        if copy_index:
            mdf.index = self.index

        return mdf


    @property
    def ticker_col_level_name(self) -> str:
        """
        Get the name of the ticker column level.

        Returns:
            str: The name of the ticker column level.
        """
        return 'ticker'
    

    @property
    def table_col_level_name(self) -> str:
        """
        Get the name of the table column level.

        Returns:
            str: The name of the table column level.
        """
        return 'table'

    
    @property
    def field_col_level_name(self) -> str:
        """
        Get the name of the field column level.

        Returns:
            str: The name of the field column level.
        """
        return 'field'
    

    @property
    def lag_col_level_name(self) -> str:
        """
        Get the name of the lag column level.

        Returns:
            str: The name of the lag column level.
        """
        return 'lag'
    

    @property
    def col_level_names(self):
        """
        Get the names of all column levels.

        Returns:
            list: The names of all column levels.
        """
        return getattr(self, '_col_level_names')
    

    @property
    def _constructor(self):
        """
        Get the constructor for creating new instances of MarketDataFrame.

        Returns:
            function: A constructor function for MarketDataFrame.
        """
        def f(*args, **kw):
            md_kwargs = {md_key: getattr(self, md_key, None) for md_key in self._metadata}
            df = MarketDataFrame(*args, **md_kwargs, **kw)
            return df
        return f
    

    @property
    def num_levels(self) -> int:
        """
        Get the number of levels in the column MultiIndex.

        Returns:
            int: The number of levels in the column MultiIndex.
        """
        if self.is_str_cols:
            self.to_tuple_cols()
            n = len(self.columns[0])
            self.to_str_cols()
            return n
        return len(self.columns[0])
    

    def set_default_metadata(self, md_key: str):
        """
        Set default metadata for a given key.

        Args:
            md_key (str): The metadata key to set.

        Raises:
            ValueError: If no setter is defined for the given metadata key.
        """
        
        if md_key == '_date_idx_mappings':
            mappings = get_date_idx_mappings(self)
            if mappings:
                i2d, d2i = mappings
                setattr(self, md_key, DateIdxMappings(i2d=i2d, d2i=d2i))
            else:
                setattr(self, md_key, None)
        
        elif md_key == '_str_tuple_col_sep':
            setattr(self, md_key, COL_LEVEL_SEP)

        elif md_key == '_nlp_module':
            setattr(self, md_key, MarketDataFrameNLP(df=self))

        elif md_key == '_col_level_names':
            setattr(self, md_key, None)
        
        else:
            raise ValueError(f'no setter defined for md_key {md_key}')
    

    # @overload
    # def __getitem__(self, key: str | tuple) -> pd.Series: # type: ignore
    #     ...

    # def __getitem__(self, key) -> pd.Series | Self:
    #     """
    #     Help out the user by intervening when they are passing column name(s) in
    #     string form when the columns are in tuple form, and vice-versa.
    #     """

    #     if isinstance(key, str) and self.is_tuple_cols:
    #         key = self._str_cname_to_tuple(key)
    #     elif isinstance(key, tuple) and self.is_str_cols:
    #         key = self._tuple_cname_to_str(key)
    #     elif isinstance(key, list) and isinstance(key[0], str) and self.is_tuple_cols:
    #         key = [self._str_cname_to_tuple(x) for x in key]
    #     elif isinstance(key, list) and isinstance(key[0], tuple) and self.is_str_cols:
    #         key = [self._tuple_cname_to_str(x) for x in key]

    #     result = super().__getitem__(key)
    #     if isinstance(result, pd.DataFrame):
    #         assert type(result) == Self
    #     return result
    

    @property
    def nlp(self) -> MarketDataFrameNLP:
        """
        Get the NLP module for this MarketDataFrame.

        Returns:
            MarketDataFrameNLP: The NLP module for this MarketDataFrame.
        """
        return getattr(self, '_nlp_module')


    @property
    def date_idx_mappings(self) -> DateIdxMappings | None:
        """
        Get the date-index mappings for this MarketDataFrame.

        Returns:
            DateIdxMappings | None: The date-index mappings, or None if not set.
        """
        return getattr(self, '_date_idx_mappings')
    

    @property
    def d2i(self) -> dict[datetime, int] | None:
        """
        Get the date-to-index mapping.

        Returns:
            dict[datetime, int] | None: The date-to-index mapping, or None if not set.
        """
        mappings = self.date_idx_mappings
        if mappings:
            return mappings.d2i
        return None
    

    @property
    def i2d(self) -> dict[int, datetime] | None:
        """
        Get the index-to-date mapping.

        Returns:
            dict[int, datetime] | None: The index-to-date mapping, or None if not set.
        """
        mappings= self.date_idx_mappings
        if mappings:
            return mappings.i2d
        return None


    @property
    def is_str_cols(self):
        """
        Check if the columns of the dataframe are currently in string form.

        Returns:
            bool: True if columns are in string form, False otherwise.
        """
        return all([isinstance(c, str) for c in self.columns])
    

    @property
    def is_tuple_cols(self):
        """
        Check if the columns of the dataframe are currently in tuple form.

        Returns:
            bool: True if columns are in tuple form, False otherwise.
        """
        return all([isinstance(c, tuple) for c in self.columns])
    

    @property
    def is_dt_index(self) -> bool:
        """
        Check if the index of the dataframe is currently in datetime form.

        Returns:
            bool: True if index is in datetime form, False otherwise.
        """
        return type(self.index) is pd.DatetimeIndex
    

    @property
    def is_int_index(self) -> bool:
        """
        Check if the index of the dataframe is currently in integer form.

        Returns:
            bool: True if index is in integer form, False otherwise.
        """
        return type(self.index) is pd.Index
    

    @property
    def _col_sep(self) -> str:
        """
        Get the string separator used when collapsing tuple columns into strings.

        Returns:
            str: The column separator.
        """
        sep = sep = self._str_tuple_col_sep
        assert isinstance(sep, str)
        return sep
    

    @property
    def tables(self) -> list[str]:
        """
        Get a list of unique table names from the column MultiIndex.

        Returns:
            list[str]: A list of unique table names.

        Raises:
            AssertionError: If the dataframe doesn't have exactly 3 column levels.
        """
        assert len(self._make_cname_tuple(self.columns[0])) == 3, 'dataframe must have 3 column levels to get table names'
        tuple_cnames = [self._make_cname_tuple(c) for c in self.columns]
        return list(set([x[1] for x in tuple_cnames]))
    

    @property
    def tickers(self) -> list[str]:
        """
        Get a list of unique ticker names from the column MultiIndex.

        Returns:
            list[str]: A list of unique ticker names.

        Raises:
            AssertionError: If the dataframe doesn't have exactly 3 column levels.
        """
        assert len(self._make_cname_tuple(self.columns[0])) == 3, 'dataframe must have 3 column levels to get ticker names'
        tuple_cnames = [self._make_cname_tuple(c) for c in self.columns]
        return list(set([x[0] for x in tuple_cnames]))
    

    @property
    def _str_cnames(self) -> list[str]:
        """
        Get a list of column names in string form.

        Returns:
            list[str]: A list of column names in string form.
        """
        return [self._make_cname_str(x) for x in self.columns]
    

    @property
    def _tuple_cnames(self) -> list[tuple]:
        """
        Get a list of column names in tuple form.

        Returns:
            list[tuple]: A list of column names in tuple form.
        """
        return [self._make_cname_tuple(x) for x in self.columns]
    

    @property
    def rows_with_missing_values(self) -> "MarketDataFrame":
        """
        Get a MarketDataFrame containing only the rows with missing values.

        Returns:
            MarketDataFrame: A new MarketDataFrame containing only the rows with missing values.
        """
        df = self[self.isnull().any(axis=1)]
        assert isinstance(df, MarketDataFrame)
        return df
    

    @property
    def numeric_cnames(self) -> list[tuple] | list[str]:
        """
        Get a list of column names for numeric columns.

        Returns:
            list[tuple] | list[str]: A list of column names for numeric columns.
        """
        return self.get_numeric_cols()


    def set_col_level_names(self, names: list[str]) -> None:
        """
        Set the names for column levels.

        Args:
            names (list[str]): A list of names for column levels.
        """
        setattr(self, '_col_level_names', names)


    def create_column_single_name(self, cname: str, series: pd.Series) -> None:
        """
        Create a new column with a single name.

        Args:
            cname (str): The name for the new column.
            series (pd.Series): The data for the new column.
        """
        null_val = ''
        tuple_cname = [null_val for _ in range(self.num_levels)]
        tuple_cname[-1] = cname
        tuple_cname = tuple(tuple_cname)
        if self.is_str_cols:
            self.to_tuple_cols()
            self[tuple_cname] = series
            self.to_tuple_cols()
            return
        self[tuple_cname] = series
        


    def add_column_level_single_value(self, val: str, name: str, level: int = 0) -> None:
        """
        Add a new level to the column index with a single value.

        Args:
            val (str): The value to add to the new level.
            name (str): The name of the new level.
            level (int, optional): The position to insert the new level. Defaults to 0 (top level).
        """

        # track whether we need to toggle back to string column form after operation
        was_str_cols = self.is_str_cols

        # perform this operation on tuple form
        self.to_tuple_cols()

        # convert columns from a list of tuples to a list of list for insertablity
        new_cols = [list(cname) for cname in self.columns]

        # insert `val` into the list form cnames
        for x in new_cols:
            x.insert(level, val)

        # convert list cnames to tuple cnames and overwrite self.columns
        new_cols = [tuple(x) for x in new_cols]
        self.columns = pd.MultiIndex.from_tuples(new_cols)

        # toggle back to string form if that was the original form
        if was_str_cols:
            self.to_str_cols()

        col_level_names = getattr(self, '_col_level_names')
        if col_level_names:
            col_level_names.insert(level, name)
            setattr(self, '_col_level_names', col_level_names)


    def _str_cname_to_tuple(self, cname: str) -> tuple:
        """
        Convert a single string column name to tuple form.

        Args:
            cname (str): The string column name to convert.

        Returns:
            tuple: The column name in tuple form.
        """
        return tuple(cname.split(self._col_sep))
    

    def _tuple_cname_to_str(self, cname: tuple) -> str:
        """
        Convert a single tuple column name to string form.

        Args:
            cname (tuple): The tuple column name to convert.

        Returns:
            str: The column name in string form.
        """
        return self._col_sep.join(cname)
    

    def _make_cname_str(self, cname: str | tuple) -> str:
        """
        Force a column name in any form into string form.

        Args:
            cname (str | tuple): The column name to convert.

        Returns:
            str: The column name in string form.
        """
        if isinstance(cname, tuple):
            return self._tuple_cname_to_str(cname)
        return cname
    

    def _make_cname_tuple(self, cname: str | tuple) -> tuple:
        """
        Force a column name in any form into tuple form.

        Args:
            cname (str | tuple): The column name to convert.

        Returns:
            tuple: The column name in tuple form.
        """
        if isinstance(cname, str):
            return self._str_cname_to_tuple(cname)
        return cname


    def to_str_cols(self):
        """
        Force dataframe into string column form (even if it is already in that form).
        """
        if self.is_str_cols:
            return
        elif self.is_tuple_cols:
            new_cols = [self._tuple_cname_to_str(cname) for cname in self.columns] # type: ignore
            self.columns = new_cols
            return
        else:
            raise ValueError(f'my columns are not of either str or tuple type')
        

    def to_tuple_cols(self):
        """
        Force dataframe into tuple column form (even if it is already in that form).
        """
        if self.is_str_cols:
            new_cols = [self._str_cname_to_tuple(cname) for cname in self.columns]
            new_col_index = pd.MultiIndex.from_tuples(new_cols)
            self.columns = new_col_index
        elif self.is_tuple_cols:
            return
        else:
            raise ValueError(f'my columns are not of either str or tuple type')
        
        col_level_names = getattr(self, '_col_level_names')
        if col_level_names:
            self.columns.names = col_level_names

        

    def to_int_index(self):
        """
        Force dataframe into integer index form (even if it is already in that form).

        Raises:
            ValueError: If the index type is not supported.
        """
        if type(self.index) is pd.Index:
            return
        elif type(self.index) is pd.DatetimeIndex:
            d2i = self.d2i
            if not d2i:
                self.set_default_metadata('_date_idx_mappings')
                return self.to_int_index()
            new_index = self.index.map(lambda x: d2i[x])
            assert type(new_index) is pd.Index
            self.index = new_index
            return
        else:
            raise ValueError(f'we do not support {type(self.index)} indexes')


    def to_dt_index(self):
        """
        Force dataframe into datetime index form (even if it is already in that form).

        Raises:
            AttributeError: If there are no index to date mappings.
            ValueError: If the index type is not supported.
        """
        if type(self.index) is pd.DatetimeIndex:
            return

        if type(self.index) is pd.Index or type(self.index) is pd.RangeIndex:
            i2d = self.i2d
            if not i2d:
                raise AttributeError('no index to date mappings; cannot convert to datetime index')
            new_index = self.index.map(lambda x: i2d[x])
            assert type(new_index) is pd.DatetimeIndex
            self.index = new_index
            return
        
        else:
            raise ValueError(f'MarketDataFrame does not support {type(self.index)} type indexes')
        
    
    def nonnull_start_end_dates_for_col(self, cname: str | tuple) -> tuple[datetime | None, datetime | None]:
        """
        Return the index values for the first non-null value and the last non-null value.

        Args:
            cname (str | tuple): The column name.

        Returns:
            tuple[datetime | None, datetime | None]: A tuple containing the start and end dates.
        """
        if self.is_int_index:
            self.to_dt_index()
            nonnull_start, nonnull_end = get_first_and_last_valid_index(self[cname])
            self.to_int_index()
        else:
            nonnull_start, nonnull_end = get_first_and_last_valid_index(self[cname])
        return nonnull_start, nonnull_end
        

    def nonnull_cols_as_of_date(self, d: str | datetime) -> list[str | tuple]:
        """
        Get a list of column names that aren't null on the given date.

        Args:
            d (str | datetime): The date to check for non-null values.

        Returns:
            list[str | tuple]: A list of column names that aren't null on the given date.
        """
        d_ = datetime_ify(d)
        assert isinstance(d_, datetime)
        
        valid_cols = []
        for cname in self.columns:
            start, _ = self.nonnull_start_end_dates_for_col(cname)
            if start is None:
                continue
            if start <= d_:
                valid_cols.append(cname)
        return valid_cols
    

    def get_numeric_cols(self, as_type: str | None = None) -> list[str] | list[tuple]:
        """
        Get a list of column names that are currently in numeric form.

        Args:
            as_type (str | None, optional): Specify whether to return column names in 'str' or 'tuple' form.
                                            If None, returns in the current form. Defaults to None.

        Returns:
            list[str] | list[tuple]: A list of numeric column names.

        Raises:
            ValueError: If an invalid value is passed for "as_type".
        """
        numeric_cols = [c for c in self.columns if pd.api.types.is_numeric_dtype(self[c])]
        if as_type:
            if as_type == 'str':
                return [self._make_cname_str(c) for c in numeric_cols]
            elif as_type == 'tuple':
                return [self._make_cname_tuple(c) for c in numeric_cols]
            else:
                raise ValueError('invalid value passed for "as_type')
        else:
            return numeric_cols
        
        
    def filter_cnames(self, *filters, how: str = 'and') -> list[str] | list[tuple]:
        """
        Helper to filter dataframe column names based on some provided filters
        and return a list matching column names.

        For each filter will check to see if the filter string exists within the column name.

        Allows you to use a single string with spaces as a query, which will be split on spaces.

        Allows you to specify whether it should be an "and" or an "or" filter. 

        Operates on lower-cased column names and filters.

        Args:
            *filters (str): a variable number of strings you want to use as filters, or a single string with spaces.
            how (str): is this an "or" filter or an "and" filter? Must be one of those two.

        Returns:
            list[str] | list[tuple]: a list of columnns matching the filter
        """

        parsed_filters = []
        for filter in filters:
            parsed_filters.extend(filter.split())

        if self.is_str_cols:
            return filter_cnames(self, *parsed_filters, how=how)
        
        else:
            # temporarily convert to string columns, get results, convert back, and return
            self.to_str_cols()
            cnames = filter_cnames(self, *parsed_filters)
            cnames = [self._str_cname_to_tuple(c) for c in cnames]
            self.to_tuple_cols()
            return cnames
        
    
    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """
        Create a MarketDataFrame from a file path.

        Args:
            path (str | Path): The path to the file.

        Returns:
            Self: A new MarketDataFrame instance.
        """
        df = pd.read_feather(path)
        return cls(df)
    
    
    def plot_column_data_starts(
            self, 
            by_ticker: bool = False, 
            by_table: bool = False, 
            cnames: list[str] | list[tuple] | None = None,
            title: str = 'Cumulative Number Columns with Data',
            ax: Axes | None = None,
            figsize: tuple[int, int] = (10,3),
            vertical_bar_location: float = 0.8
        ) -> Axes:
        """
        Utility plotting method to visualize when data starts in time.

        This method provides four modes of operation:
        1. Visualize all data at once. (by_ticker=False, by_table=False)
        2. Visualize by ticker. (by_ticker=True, by_table=False)
        3. Visualize by table. (by_ticker=False, by_table=True)
        4. Visualize by ticker-table combinations (by_ticker=True, by_table=True)

        For the non-default modes, it will filter column names and call itself recursively.

        The method plots an adjustable vertical bar at the date where `vertical_bar_location` percentage of
        columns have data.

        Args:
            by_ticker (bool, optional): If True, visualize data by ticker. Defaults to False.
            by_table (bool, optional): If True, visualize data by table. Defaults to False.
            cnames (list[str] | list[tuple] | None, optional): Optionally specify a list of column names
                to filter down to for this plot. Main use case is recursive calls. Defaults to None.
            title (str, optional): Title of the plot. Defaults to 'Cumulative Number Columns with Data'.
            ax (Axes | None, optional): Matplotlib Axes object to plot on. If None, a new figure is created. Defaults to None.
            figsize (tuple[int, int], optional): Figure size for new plots. Defaults to (10,3).
            vertical_bar_location (float): Percentage of columns at which to plot a vertical bar. Defaults to 0.8.

        Returns:
            Axes: The matplotlib Axes object containing the plot.
        """

        if by_ticker and by_table:

            tickers = self.tickers
            tables = self.tables

            ticker_table_tuples = list(itertools.product(tickers, tables))
            
            fig, axs = plt.subplots(
                    len(ticker_table_tuples), 
                    1, 
                    sharex=True, 
                    figsize=(10, len(ticker_table_tuples)*1.5)
                )
            
            for (ticker, table), ax in zip(ticker_table_tuples, axs.ravel()):

                tuple_cnames = [self._make_cname_tuple(x) for x in self.columns]
                tuple_cnames = [x for x in tuple_cnames if x[0] == ticker and x[1] == table]
                
                _ = self.plot_column_data_starts(
                    cnames=tuple_cnames, 
                    title=f'{ticker} | {table}',
                    ax=ax
                )
            
            fig.suptitle('Cumulative Number of Columns with Data')
            plt.tight_layout()

            return axs

        elif by_ticker:

            tickers = self.tickers

            fig, axs = plt.subplots(
                    len(tickers), 
                    1, 
                    sharex=True, 
                    figsize=(10, len(tickers)*1.5)
                )
            
            for ticker, ax in zip(tickers, axs.ravel()):

                ticker_cnames = [x for x in self.columns if self._make_cname_tuple(x)[0] == ticker]
                
                _ = self.plot_column_data_starts(
                    cnames=ticker_cnames, 
                    title=ticker,
                    ax=ax
                )
            
            fig.suptitle('Cumulative Number of Columns with Data')
            plt.tight_layout()

            return axs

        elif by_table:

            tables = self.tables
            
            fig, axs = plt.subplots(
                    len(tables), 
                    1, 
                    sharex=True, 
                    figsize=(10, len(tables)*1.5)
                )
            
            for table, ax in zip(tables, axs.ravel()):

                ticker_cnames = [x for x in self.columns if self._make_cname_tuple(x)[1] == table]
                
                _ = self.plot_column_data_starts(
                    cnames=ticker_cnames, 
                    title=table,
                    ax=ax
                )
            
            fig.suptitle('Cumulative Number of Columns with Data')
            plt.tight_layout()

            return axs


        else:
            cnames_ = cnames if cnames else self.columns
            counts = defaultdict(lambda: 0)
            for cname in cnames_:
                start, _ = self.nonnull_start_end_dates_for_col(cname)
                counts[start] += 1
            counts = pd.DataFrame(counts.items(), columns=['start', 'n_cols'])
            counts = counts.set_index('start').sort_index()
            counts['n_cols_cum'] = counts['n_cols'].cumsum()
            counts['n_cols_cum_norm'] = counts['n_cols_cum'] / counts['n_cols_cum'].max()

            vertical_bar_date = counts[counts['n_cols_cum_norm'] >= vertical_bar_location].index[0]

            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            assert ax is not None

            _ = counts['n_cols_cum'].plot(
                ax=ax, 
                linestyle='dotted', 
                linewidth=1, 
                color='grey',
            )
            _ = ax.scatter(
                x=counts.index, 
                y=counts['n_cols_cum'],
                alpha=0.5,
                color='blue',
                edgecolors='none',
                marker='.',
                s=100,
            )
            _ = ax.axvline(x=vertical_bar_date, color='red', linewidth=2, linestyle='--')
            _ = ax.text(
                x=0.02, 
                y=0.95, 
                s=f'{int(vertical_bar_location*100)}% @ {vertical_bar_date.date()}',
                verticalalignment='top',
                horizontalalignment='left',
                transform=ax.transAxes,
                color='red',
                fontsize=10,
            )
            _ = ax.grid(True, which='major', axis='x', linewidth=0.5, color='lightgray', alpha=0.5)
            _ = ax.set_title(title)

            return ax





    # @property
    # def _constructor_sliced(self):
    #     def f(*args, **kw):
    #         df = MarketDataFrame(*args, **kw)
    #         self._copy_metadata(df)
    #         return df
    #     return f