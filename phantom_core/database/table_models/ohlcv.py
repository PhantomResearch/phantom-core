from sqlalchemy import Column, Integer, Float, DateTime, String, BigInteger
import pandas as pd
from sqlalchemy.orm import DeclarativeBase
from enum import Enum

class Base(DeclarativeBase):
    pass

class BaseOHLCV(Base):
    __abstract__ = True
    
    # Composite primary key of timestamp and symbol
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(12), primary_key=True)
    
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    vwap = Column(Float)
    transactions = Column(Integer)
    
    @classmethod
    def from_polygon_agg(cls, agg, symbol: str) -> "BaseOHLCV":   # TODO: this should probably be moved eventually
        """Convert a Polygon Agg object to a Bar"""
        return cls(
            timestamp=pd.Timestamp(agg.timestamp, unit='ms').tz_localize('UTC'),
            symbol=symbol,
            open=agg.open,
            high=agg.high,
            low=agg.low,
            close=agg.close,
            volume=agg.volume,
            vwap=agg.vwap,
            transactions=agg.transactions
        )

class OHLCV1M(BaseOHLCV):
    __tablename__ = "ohlcv_1m"

class OHLCV5M(BaseOHLCV):
    __tablename__ = "ohlcv_5m"