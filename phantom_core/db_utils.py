from tracemalloc import reset_peak
import pandas as pd
from sqlalchemy import create_engine, text, Result, Engine, inspect
from sqlmodel import Session
from contextlib import contextmanager

from .constants import STONKS_DATABASE_URL, PERIOD_CNAME
from .market_dataframe import MarketDataFrame
from .datasource import SourceTable


@contextmanager
def ensure_session(engine: Engine, session: Session | None = None, commit: bool = False):
    """
    Provides a session, either using existing one or creating new one.
    
    Notes:
        - Currently mainly used in trading system (eg, OMS and AMS).
    """
    if session is not None:
        yield session
    else:
        with Session(engine) as new_session:
            yield new_session
            if commit:
                new_session.commit()


def orm_statement_to_sql(stmt) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


def get_stonks_db_engine() -> Engine | None:
    """
    Create and return a SQLAlchemy engine for the Stonks database.

    Returns:
        Engine: A SQLAlchemy engine connected to the Stonks database.
    """
    if not STONKS_DATABASE_URL:
        return None
    return create_engine(STONKS_DATABASE_URL)


STONKS_ENGINE = get_stonks_db_engine()    # TODO

def get_engine() -> Engine:
    if not STONKS_ENGINE:
        raise RuntimeError('STONKS database not accessible in this environment')
    return STONKS_ENGINE


def runquery(sql: str) -> pd.DataFrame:
    """
    Execute a SQL query and return the result as a pandas DataFrame.

    Args:
        sql (str): The SQL query to execute.

    Returns:
        pd.DataFrame: The result of the SQL query as a pandas DataFrame.
    """
    return pd.read_sql(sql, get_engine())


def execute_statement(sql: str) -> Result:
    """
    Execute a SQL statement and commit the changes.

    Args:
        sql (str): The SQL statement to execute.

    Returns:
        Result: The result of the SQL statement execution.
    """
    with get_engine().connect() as con:
        result = con.execute(text(sql))
        con.commit()
    return result


def list_tables() -> pd.DataFrame:
    """
    List all tables in the database, excluding system schemas.

    Returns:
        pd.DataFrame: A DataFrame containing table_schema and table_name for all non-system tables.
    """
    sql = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE'
        AND table_schema not in ('information_schema', 'pg_catalog')
    ORDER BY table_schema, table_name;
    """
    return runquery(sql)


def listtables(engine: Engine) -> list[str]:
    insp = inspect(engine)
    return insp.get_table_names()


def drop_table(tablename: str) -> None:
    """
    Drop a table from the database.

    Args:
        tablename (str): The name of the table to drop.
    """
    sql = f'DROP TABLE {tablename}'
    with get_engine().connect() as con:
        con.execute(text(sql))
        con.commit()


def get_table_ticker_df(
        table: SourceTable | str, 
        ticker: str, period_cname: str = PERIOD_CNAME, 
        start_ts: pd.Timestamp | None = None,
        end_ts: pd.Timestamp | None = None,
    ) -> MarketDataFrame:
    """
    Get a DataFrame for a table-ticker pair, with optional start and end timestamps.

    Args:
        table (SourceTable | str): The table name or SourceTable object.
        ticker (str): The ticker symbol.
        period_cname (str): The name of the period column.
        start_ts (pd.Timestamp | None): The start timestamp.
        end_ts (pd.Timestamp | None): The end timestamp.
    """

    # Get the table name from SourceTable object or use the provided string
    table_name = table.db_name if isinstance(table, SourceTable) else table

    # Construct WHERE clauses for start and end timestamps, defaulting to True if not provided
    start_where_clause = f"{period_cname} >= '{start_ts}'" if start_ts else "True=True"
    end_where_clause = f"{period_cname} <= '{end_ts}'" if end_ts else "True=True"

    # Construct the SQL query
    sql = f"""
    SELECT *
    FROM {table_name}
    WHERE ticker='{ticker}'
        AND {start_where_clause}
        AND {end_where_clause}
    ORDER BY {period_cname}
    """
    # Execute the query and get the result as a DataFrame
    df = runquery(sql)

    # Convert the period column to datetime
    df[period_cname] = pd.to_datetime(df[period_cname])

    # Add a 'table' column with the table name
    df['table'] = table_name

    # Set the period column as index and sort
    df = df.set_index(period_cname).sort_index()

    # Convert to MarketDataFrame
    df = MarketDataFrame(df)
    
    # Set column level names
    df.set_col_level_names([df.field_col_level_name])

    # Return the final MarketDataFrame
    return MarketDataFrame(df)


def table_exists(table_name: str) -> bool:
    """
    Check if a table exists in the database.

    Args:
        table_name (str): The name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    sql = f"""
    SELECT EXISTS (
    SELECT 1 
    FROM information_schema.tables 
    WHERE table_name = '{table_name}'
    );
    """
    return bool(runquery(sql).iloc[0,0])


def get_latest_ticker_table_timestamp_from_db(table: str | SourceTable, ticker: str) -> pd.Timestamp | None:
    """
    Get the latest timestamp for a specific ticker in a table.

    Args:
        table (str | SourceTable): The table name or SourceTable object.
        ticker (str): The ticker symbol.

    Returns:
        pd.Timestamp | None: The latest timestamp or None if not found.
    """
    table_name = table.db_name if isinstance(table, SourceTable) else table
    
    if not table_exists(table_name=table_name):
        return None
    
    sql = f"""
    SELECT max({PERIOD_CNAME}) as {PERIOD_CNAME}
    FROM {table_name}
    WHERE ticker='{ticker}'
    """

    return runquery(sql).iloc[0][PERIOD_CNAME]


def list_users() -> pd.DataFrame:
    """
    List all users in the database.

    Returns:
        pd.DataFrame: A DataFrame containing information about all users in the database.
    """
    return runquery('SELECT * FROM pg_roles')


def inspect_user_permissions(user: str) -> pd.DataFrame:
    """
    Inspect permissions for a specific user.

    Args:
        user (str): The username to inspect permissions for.

    Returns:
        pd.DataFrame: A DataFrame containing the user's permissions.
    """
    sql = f"""
    SELECT grantee, privilege_type 
    FROM information_schema.role_table_grants 
    WHERE grantee='{user}';
    """
    return runquery(sql)


def get_database_size():
    """
    Retrieve and display the size of the current database and its tables.

    This function queries the database to get the total size of the current database,
    the combined size of all tables, and the sizes of the top 10 largest tables.
    It then prints this information to the console.

    The function uses two SQL queries:
    1. To get the overall database size and total size of all tables.
    2. To get the sizes of individual tables, ordered by size descending.

    Returns:
        None

    Prints:
        - Total Database Size
        - Size of All Tables
        - Top 10 Largest Tables with their respective sizes
    """
    # Query to get the size of the database and all tables
    db_size_query = """
        SELECT pg_size_pretty(pg_database_size(current_database())) as db_size,
               pg_size_pretty(sum(pg_total_relation_size(c.oid))) as tables_size
        FROM pg_class c
        LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND c.relkind IN ('r', 'p')
    """
    result = runquery(db_size_query)
    if result.empty:
        print("No results returned from the database size query.")
        return
    
    # Query to get sizes of individual tables
    table_sizes_query = """
        SELECT relname as table_name,
               pg_size_pretty(pg_total_relation_size(c.oid)) as table_size
        FROM pg_class c
        LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND c.relkind IN ('r', 'p')
        ORDER BY pg_total_relation_size(c.oid) DESC
        LIMIT 10
    """
    table_sizes = runquery(table_sizes_query)
    
    if not result.empty:
        print(f"Total Database Size: {result.iloc[0]['db_size']}")
        print(f"Size of All Tables: {result.iloc[0]['tables_size']}")
    
    if not table_sizes.empty:
        print("\nTop 10 Largest Tables:")
        for _, row in table_sizes.iterrows():
            print(f"{row['table_name']}: {row['table_size']}")
    else:
        print("No table size information available.")
