from enum import Enum

class DatabaseName(str, Enum):
    RAW_INGEST = "raw_ingest"
    LOGGING = "logging_db"