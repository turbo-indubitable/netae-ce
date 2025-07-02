# networkae/utils/timesummary.py

import pandas as pd
import logging

# Reuse or import from a central location if it's defined there
class NullLogger:
    def log(self, level, msg): pass
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def log_with_tag(logger, level, tag, msg):
    if not logger:
        print(f"[{tag}] {msg}")
        return
    logger.log(level, f"[{tag}] {msg}")


def print_timestamp_summary(df: pd.DataFrame, column: str = "timestamp", logger=None) -> None:
    """
    Log a human-readable summary of the timestamp range in the given DataFrame.

    Args:
        df: Input DataFrame with a timestamp column.
        column: Name of the timestamp column (default: "timestamp").
        logger: Optional logger to send output to. Falls back to print().
    """
    logger = logger or NullLogger()

    if column not in df.columns:
        log_with_tag(logger, logging.WARNING, "TimeSummary", f"Timestamp column '{column}' not found.")
        return

    ts_col = df[column]
    try:
        # Convert to datetime (assume float32/float64 as scaled Unix time in seconds)
        if pd.api.types.is_numeric_dtype(ts_col):
            ts_col = pd.to_datetime(ts_col * 1e9, errors='coerce')  # Convert to ns
        elif not pd.api.types.is_datetime64_any_dtype(ts_col):
            ts_col = pd.to_datetime(ts_col, errors='coerce')

        ts_col = ts_col.dropna()
        if ts_col.empty:
            log_with_tag(logger, logging.WARNING, "TimeSummary", f"No valid datetime values in column '{column}'")
            return

        start_time = ts_col.min()
        end_time = ts_col.max()
        duration = end_time - start_time

        def pretty_duration(delta: pd.Timedelta) -> str:
            days = delta.days
            seconds = delta.seconds
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            parts = []
            if days > 0:
                parts.append(f"{days} days")
            if hours > 0:
                parts.append(f"{hours} hours")
            if minutes > 0:
                parts.append(f"{minutes} minutes")
            return ", ".join(parts) or "less than a minute"

        log_with_tag(logger, logging.INFO, "TimeSummary", f"Time range detected: {pretty_duration(duration)}")
        log_with_tag(logger, logging.INFO, "TimeSummary", f"First timestamp: {start_time}")
        log_with_tag(logger, logging.INFO, "TimeSummary", f"Last timestamp:  {end_time}")

    except Exception as e:
        log_with_tag(logger, logging.WARNING, "TimeSummary", f"Failed to process timestamp column '{column}': {e}")