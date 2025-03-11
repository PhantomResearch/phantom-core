from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Set up logger
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(OperationalError),
    reraise=True
)
def get_postgres_engine(
    database: str,
    env_prefix: str = "PHANTOM_POSTGRES",
    host_var: str | None = None,
    port_var: str | None = None,
    username_var: str | None = None,
    password_var: str | None = None,
    default_port: int = 5432,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    connect_timeout: int = 10
) -> Engine:
    """
    Create a PostgreSQL SQLAlchemy engine using database credentials from environment variables.
    Includes retry logic, connection pooling, and connection testing.
    
    Args:
        database: Database name (required)
        env_prefix: Prefix for environment variables (default: "PHANTOM")
        host_var: Environment variable name for host (default: "{env_prefix}_HOST")
        port_var: Environment variable name for port (default: "{env_prefix}_PORT")
        username_var: Environment variable name for username (default: "{env_prefix}_USER")
        password_var: Environment variable name for password (default: "{env_prefix}_PASSWORD")
        default_port: Default port to use if port_var is not set (default: 5432)
        pool_size: Size of the connection pool (default: 5)
        max_overflow: Maximum overflow connections (default: 10)
        pool_timeout: Timeout for getting a connection from pool in seconds (default: 30)
        connect_timeout: Timeout for establishing connection in seconds (default: 10)
        
    Returns:
        SQLAlchemy engine for PostgreSQL
        
    Raises:
        ValueError: If both env_prefix and any specific environment variable names are provided
    """
    # Validate that user is not mixing env_prefix with specific variable names
    specific_vars = [host_var, port_var, username_var, password_var]
    if env_prefix and any(specific_vars):
        raise ValueError(
            "Cannot specify both env_prefix and specific environment variable names. "
            "Either use env_prefix alone, or specify all variable names individually."
        )
    
    # Set default variable names if not provided
    if env_prefix:
        host_var = f"{env_prefix}_HOST"
        port_var = f"{env_prefix}_PORT"
        username_var = f"{env_prefix}_USER"
        password_var = f"{env_prefix}_PASSWORD"
    else:
        # Ensure all specific variable names are provided if env_prefix is not used
        if not all(specific_vars):
            raise ValueError(
                "If env_prefix is not provided, all specific environment variable names "
                "(host_var, port_var, username_var, password_var) must be specified."
            )
    
    # Get values from environment variables
    host = os.environ.get(host_var or "", "localhost")
    port = os.environ.get(port_var or "", str(default_port))
    username = os.environ.get(username_var or "", "")
    password = os.environ.get(password_var or "", "")
    
    logger.info(f"Creating database engine for database: {database}")
    
    # Construct connection string
    if username and password:
        connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        log_connection = f"postgresql://{username}:***@{host}:{port}/{database}"
    else:
        connection_string = f"postgresql://{host}:{port}/{database}"
        log_connection = connection_string
    
    logger.debug(f"Using connection string: {log_connection}")
    
    # Add connection pooling and timeout settings
    logger.info("Configuring engine with connection pooling")
    engine = create_engine(
        connection_string,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_pre_ping=True,  # Verify connection before using from pool
        connect_args={
            "connect_timeout": connect_timeout  # Seconds to wait for connection
        }
    )
    
    try:
        # Test connection
        logger.info("Testing database connection...")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection test successful")
        return engine
    except OperationalError:
        # Log the error but re-raise it for the retry decorator to catch
        logger.error("Database operational error (will retry)")
        raise
    except Exception as e:
        # For other exceptions, log and re-raise
        logger.error(f"Database connection failed: {str(e)}")
        raise