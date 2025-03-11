import pytest
import pandas as pd
from datetime import datetime

from phantom_core.market_calendar import get_market_days


def test_timestamps_must_have_timezone():
    """Test that timestamps must have timezone information."""
    # Create timestamps without timezone
    start_ts = pd.Timestamp('2025-01-01')
    end_ts = pd.Timestamp('2025-01-05')
    
    # Test with start_ts missing timezone
    with pytest.raises(ValueError, match="start_ts must have timezone information"):
        get_market_days(start_ts, end_ts.tz_localize('UTC'))
    
    # Test with end_ts missing timezone
    with pytest.raises(ValueError, match="end_ts must have timezone information"):
        get_market_days(start_ts.tz_localize('UTC'), end_ts)


def test_timestamps_must_have_same_timezone():
    """Test that both timestamps must have the same timezone."""
    # Create timestamps with different timezones
    start_ts = pd.Timestamp('2025-01-01').tz_localize('UTC')
    end_ts = pd.Timestamp('2025-01-05').tz_localize('US/Eastern')
    
    with pytest.raises(ValueError, match="start_ts and end_ts must have the same timezone"):
        get_market_days(start_ts, end_ts)


def test_returned_timestamps_have_same_timezone():
    """Test that returned timestamps have the same timezone as inputs."""
    # Test with UTC timezone
    start_ts = pd.Timestamp('2025-01-01').tz_localize('UTC')
    end_ts = pd.Timestamp('2025-01-05').tz_localize('UTC')
    
    days = get_market_days(start_ts, end_ts)
    
    # Check that we got some results
    assert len(days) > 0
    
    # Check that all returned timestamps have the same timezone as input
    for day in days:
        assert str(day.tzinfo) == str(start_ts.tzinfo)
    
    # Test with US/Eastern timezone
    start_ts_eastern = pd.Timestamp('2025-01-01').tz_localize('US/Eastern')
    end_ts_eastern = pd.Timestamp('2025-01-05').tz_localize('US/Eastern')
    
    days_eastern = get_market_days(start_ts_eastern, end_ts_eastern)
    
    # Check that all returned timestamps have the same timezone as input
    for day in days_eastern:
        assert str(day.tzinfo) == str(start_ts_eastern.tzinfo)


def test_timezone_consistency():
    """Test that results are consistent across different timezones."""
    # Same date range in different timezones
    start_date = '2025-01-01'
    end_date = '2025-01-10'
    
    # Get results in UTC
    days_utc = get_market_days(
        pd.Timestamp(start_date).tz_localize('UTC'),
        pd.Timestamp(end_date).tz_localize('UTC')
    )
    
    # Get results in US/Eastern
    days_eastern = get_market_days(
        pd.Timestamp(start_date).tz_localize('US/Eastern'),
        pd.Timestamp(end_date).tz_localize('US/Eastern')
    )
    
    # Convert all to naive timestamps for comparison
    days_utc_naive = [day.tz_localize(None) for day in days_utc]
    days_eastern_naive = [day.tz_localize(None) for day in days_eastern]
    
    # The dates should be the same regardless of timezone
    assert days_utc_naive == days_eastern_naive


def test_weekend_exclusion():
    """Test that weekends are excluded from market days."""
    # January 2025: 
    # - 4-5 is a weekend (Sat-Sun)
    # - 11-12 is a weekend (Sat-Sun)
    start_ts = pd.Timestamp('2025-01-01').tz_localize('UTC')
    end_ts = pd.Timestamp('2025-01-15').tz_localize('UTC')
    
    days = get_market_days(start_ts, end_ts)
    
    # Convert to dates for easier checking
    dates = [day.date() for day in days]
    
    # Check that weekends are not included
    weekend_dates = [
        datetime(2025, 1, 4).date(),  # Saturday
        datetime(2025, 1, 5).date(),  # Sunday
        datetime(2025, 1, 11).date(),  # Saturday
        datetime(2025, 1, 12).date(),  # Sunday
    ]
    
    for weekend_date in weekend_dates:
        assert weekend_date not in dates, f"Weekend {weekend_date} should not be in market days"


def test_holiday_exclusion():
    """Test that holidays are excluded from market days."""
    # Test for common US holidays in 2025
    holidays = [
        # New Year's Day
        (pd.Timestamp('2025-01-01').tz_localize('UTC'), "New Year's Day"),
        
        # Martin Luther King Jr. Day (3rd Monday in January)
        (pd.Timestamp('2025-01-20').tz_localize('UTC'), "Martin Luther King Jr. Day"),
        
        # Presidents' Day (3rd Monday in February)
        (pd.Timestamp('2025-02-17').tz_localize('UTC'), "Presidents' Day"),
        
        # Good Friday (April 18, 2025)
        (pd.Timestamp('2025-04-18').tz_localize('UTC'), "Good Friday"),
        
        # Memorial Day (Last Monday in May)
        (pd.Timestamp('2025-05-26').tz_localize('UTC'), "Memorial Day"),
        
        # Independence Day (July 4, observed on Friday if Saturday, Monday if Sunday)
        (pd.Timestamp('2025-07-04').tz_localize('UTC'), "Independence Day"),
        
        # Labor Day (1st Monday in September)
        (pd.Timestamp('2025-09-01').tz_localize('UTC'), "Labor Day"),
        
        # Thanksgiving (4th Thursday in November)
        (pd.Timestamp('2025-11-27').tz_localize('UTC'), "Thanksgiving"),
        
        # Christmas
        (pd.Timestamp('2025-12-25').tz_localize('UTC'), "Christmas"),
    ]
    
    for holiday_ts, holiday_name in holidays:
        # Check each holiday individually
        days = get_market_days(holiday_ts, holiday_ts)
        assert len(days) == 0, f"{holiday_name} ({holiday_ts.date()}) should not be a market day"
