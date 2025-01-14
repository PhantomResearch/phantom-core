import functools
import json
import logging
from logging import Logger
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd
from pytz import timezone
from contextlib import contextmanager, redirect_stdout
from typing import Any, Literal, TextIO

from .constants import LOGGING_TIME_ZONE


class LoggerStream(TextIO):
    """
    A custom stream class that writes to a logger instead of standard output.

    This class implements a subset of the TextIO interface, specifically the
    write and flush methods, to redirect output to a logger at the INFO level.

    Attributes:
        logger (Logger): The logger object to which the output will be written.
        linebuf (str): A buffer to store incomplete lines.
    """

    def __init__(self, logger):
        """
        Initialize the LoggerStream with a logger.

        Args:
            logger (Logger): The logger object to use for output.
        """
        self.logger = logger
        self.linebuf = ""

    def write(self, buf):
        """
        Write the given buffer to the logger, line by line.

        Args:
            buf (str): The string buffer to write.
        """
        for line in buf.rstrip().splitlines():
            self.logger.info(line.rstrip())

    def flush(self):
        """
        Flush any remaining content in the line buffer to the logger.
        """
        if self.linebuf:
            self.logger.info(self.linebuf.rstrip())
            self.linebuf = ""


def get_logger(name: str) -> Logger:
    """
    Get or create a logger with the given name and configure it with a StreamHandler.

    This function creates a new logger if one doesn't exist, or returns an existing logger.
    It also ensures that the logger has a StreamHandler with a custom formatter that includes
    the timezone specified in LOGGING_TIME_ZONE.

    Args:
        name (str): The name of the logger, typically __name__.

    Returns:
        Logger: A configured logger object.

    Example:
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    
    # Only add handler if the logger doesn't already have handlers
    if not logger.handlers:
        formatter = logging.Formatter(
            f'%(levelname)s | %(asctime)s.%(msecs)03d ({LOGGING_TIME_ZONE}) | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        formatter.converter = lambda *args: datetime.now(tz=timezone(LOGGING_TIME_ZONE)).timetuple()

        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    logger.propagate = True
    
    return logger


@contextmanager
def log_stdout(logger: Logger):
    """
    A context manager that captures the stdout output and logs it in real-time using the provided logger.

    Args:
        logger (Logger): The logger object to use for logging the captured output.
    """
    logger_stream = LoggerStream(logger)
    with redirect_stdout(logger_stream):
        yield
    logger_stream.flush()


def capture_output_to_logger(logger):
    """
    A decorator that captures the stdout output of a function and logs it in real-time using the provided logger.

    This decorator redirects the standard output of the decorated function to a custom LoggerStream.
    The output is logged immediately using the provided logger at the INFO level.

    Args:
        logger (logging.Logger): The logger object to use for logging the captured output.

    Returns:
        function: A decorator function that wraps the original function.

    Example:
        @capture_output_to_logger(my_logger)
        def my_function():
            print("This will be captured and logged immediately")
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with log_stdout(logger):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


def construct_header(text: str, width: int = 80, horiz_char: str = '=', vert_char: str = '|') -> str:
    """
    Construct a header string for console output.

    Args:
        text (str): The text to be displayed in the header.
        width (int, optional): The total width of the header. Defaults to 80.
        horiz_char (str, optional): The character used for horizontal lines. Defaults to '='.
        vert_char (str, optional): The character used for vertical lines. Defaults to '|'.

    Returns:
        str: A formatted header string.
    """
    header = f"\n{horiz_char * width}\n"
    header += f"{vert_char} {text.center(width - 4)} {vert_char}\n"
    header += f"{horiz_char * width}\n"
    return header


def log_df(df: pd.DataFrame, logger: Logger, header: str | None = None):
    
    if header is not None:
        _msgs = construct_header(header).split('\n')
    else:
        _msgs = []
    
    _msgs.extend(df.to_string().split('\n'))
    for msg in _msgs:
        logger.info(msg)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


class JsonLinesFormatter(logging.Formatter):

    def __init__(self):
        self.attrs_to_log = {
            'name': '_logger_name',
            'levelname': '_logging_level',
            'module': '_module_name',
            'funcName': '_func_name',
            'lineno': '_lineno',
        }
        return super().__init__()


    def format(self, record):

        assert isinstance(record.msg, dict)

        for attr_name, new_attr_name in self.attrs_to_log.items():
            record.msg[new_attr_name] = getattr(record, attr_name)

        record.msg['_timestamp'] = pd.Timestamp.utcnow().isoformat()

        record.msg = json.dumps(record.msg, cls=DateTimeEncoder)

        return super().format(record)


def get_data_logger(name: str, filepath: Path | str) -> Logger:

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    formatter = JsonLinesFormatter()
    
    handler = logging.FileHandler(filepath)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


class LoggingMixin:
    """
    A mixin class that provides logging functionality to other classes.

    This mixin class provides methods for logging messages with different levels
    (info, debug, warning, error) and for logging DataFrames.

    Attributes:
        logger (Logger): The logger object to use for logging messages.
    """
    
    logger: Logger
    data_loggers: dict[str, Logger] | None = None


    def log(
            self, 
            msg: Any, 
            *,
            level: Literal['info', 'debug', 'warning', 'error'] = 'info',
            use_header: bool = False,
            **kwargs: dict[str, Any],
        ):
        """
        Log a message with the specified level.

        Args:
            msg (Any): The message to log.
            level (Literal['info', 'debug', 'warning', 'error'], optional): The logging level. Defaults to 'info'.
            use_header (bool, optional): Whether to wrap the message in a header. Defaults to False.
        """

        if use_header:
            _msgs = construct_header(str(msg)).split('\n')
        else:
            _msgs = [msg]
            
        for msg in _msgs:
            getattr(self.logger, level)(str(msg))

        if kwargs is not None and self.data_loggers is not None:
            for logger_key, logger_msg in kwargs.items():
                data_logger = self.data_loggers.get(logger_key)
                if data_logger is not None:
                    getattr(data_logger, level)(logger_msg)


    def log_df(self, df: pd.DataFrame, msg: str | None = None, use_header: bool = True):
        """
        Log a DataFrame, optionally with a message header.

        Args:
            df (pd.DataFrame): The DataFrame to log.
            msg (str | None, optional): An optional message to log before the DataFrame. Defaults to None.
            use_header (bool, optional): Whether to wrap the message in a header. Defaults to True.
        """

        if msg is not None:
            self.log(msg, use_header=use_header)

        log_df(df, self.logger)

    
    def init_data_loggers(self, data_logger_specs: dict[str, str | Path]) -> None:

        _loggers = dict()
        for logger_key, file_path in data_logger_specs.items():
            _loggers[logger_key] = get_data_logger(logger_key, file_path)

        self.data_loggers = _loggers
        


    
