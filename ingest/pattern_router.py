# networkae/ingest/pattern_router.py

import hashlib
import logging
from typing import List, Dict, Tuple, Generator, Union, Callable, Optional

import pandas as pd


class NullLogger:
    def log(self, level, msg): pass
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def log_with_tag(logger: Optional[logging.Logger], level: int, tag: str, msg: str) -> None:
    if not logger or not hasattr(logger, "log"):
        print(f"[{tag}] {msg}")
    else:
        logger.log(level, f"[{tag}] {msg}")


def parse_pattern_filters(
    patterns: List[Dict[str, str]],
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Union[str, Callable[[pd.DataFrame], pd.Index]]]]:
    logger = logger or NullLogger()
    parsed = []

    for pattern in patterns:
        expr = pattern['filter_expr']
        pattern_id = pattern['pattern_id']
        if pattern_id == "unmatched":
            log_with_tag(logger, logging.INFO, "Pipeline", f"Skipping unmatched traffic — no model created.")
            continue

        ae_id = hashlib.sha256(pattern_id.encode()).hexdigest()[:16]

        def make_filter(expr: str) -> Callable[[pd.DataFrame], pd.Index]:
            return lambda df: df.query(expr, engine='python').index

        parsed.append({
            'pattern_id': pattern_id,
            'filter_expr': expr,
            'ae_id': ae_id,
            'match_fn': make_filter(expr)
        })

        log_with_tag(logger, logging.DEBUG, "PatternRouter", f"Parsed pattern '{pattern_id}' with filter: {expr}")

    return parsed


def assign_time_bins(
    df: pd.DataFrame,
    timestamp_col: str,
    bin_size: int,
    grace_period: int
) -> pd.DataFrame:
    df['_epoch'] = pd.to_datetime(df[timestamp_col], unit='s').astype('int64') // 10 ** 9
    df['_bin_id'] = ((df['_epoch'] - (df['_epoch'] % bin_size)) // bin_size)
    return df


def route_and_bin(
    df: pd.DataFrame,
    parsed_patterns: List[Dict[str, Union[str, Callable[[pd.DataFrame], pd.Index]]]],
    bin_size_seconds: int = 15,
    stream: bool = True,
    logger: Optional[logging.Logger] = None
) -> Generator[Tuple[str, str, pd.Timestamp, pd.DataFrame], None, None]:
    logger = logger or NullLogger()
    total_rows = len(df)
    seen_indices = set()

    log_with_tag(logger, logging.DEBUG, "PatternRouter", "[DEBUG] Sample rows:\n" + str(df.head(3)))
    log_with_tag(logger, logging.DEBUG, "PatternRouter", f"dtypes:\n{df.dtypes}")
    log_with_tag(logger, logging.DEBUG, "PatternRouter", "[DEBUG] Protocol distribution:\n" + str(df['protocol'].value_counts(dropna=False)))
    log_with_tag(logger, logging.DEBUG, "PatternRouter", "[DEBUG] Ports distribution:\n" +
                 str(df['src_port'].value_counts(dropna=False).head(5)) + "\n" +
                 str(df['dst_port'].value_counts(dropna=False).head(5)))

    for pattern in parsed_patterns:
        pattern_id: str = pattern['pattern_id']  # type: ignore
        ae_id: str = pattern['ae_id']            # type: ignore
        filter_expr: str = pattern['filter_expr']  # type: ignore
        match_fn: Callable[[pd.DataFrame], pd.Index] = pattern['match_fn']  # type: ignore

        log_with_tag(logger, logging.INFO, "PatternRouter", f"[Routing] Pattern '{pattern_id}' → filter: {filter_expr}")

        try:
            matched_indices = match_fn(df)
            matched = df.loc[matched_indices]
            log_with_tag(logger, logging.DEBUG, "PatternRouter", f"[DEBUG] Fields in matched rows: {list(matched.columns)}")
        except Exception as e:
            log_with_tag(logger, logging.ERROR, "PatternRouter", f"[ERROR] Failed to apply filter for pattern '{pattern_id}': {e}")
            continue

        log_with_tag(logger, logging.INFO, "PatternRouter", f"[Routing] Pattern '{pattern_id}' matched {len(matched)} rows")
        if matched.empty:
            continue

        seen_indices.update(matched.index)

        matched = matched.copy()
        if not pd.api.types.is_datetime64_any_dtype(matched['timestamp']):
            if matched['timestamp'].max() > 1e12:  # nanoseconds
                temp_ts = pd.to_datetime(matched['timestamp'], unit='ns')
            else:
                temp_ts = pd.to_datetime(matched['timestamp'], unit='s')
        else:
            temp_ts = matched['timestamp']

        matched['bin_start'] = temp_ts.dt.floor(f"{bin_size_seconds}s")

        log_with_tag(logger, logging.DEBUG, "PatternRouter", f"[DEBUG] matched['timestamp'] dtype: {matched['timestamp'].dtype}")
        log_with_tag(logger, logging.DEBUG, "PatternRouter", f"[DEBUG] Sample timestamps: {matched['timestamp'].head(3).tolist()}")

        for bin_start, bin_df in matched.groupby("bin_start"):
            log_with_tag(logger, logging.INFO, "PatternRouter", f"[Binning] {pattern_id} | {bin_start} | rows: {len(bin_df)}")
            assert 'src_ip' in bin_df.columns and 'dst_ip' in bin_df.columns, "[ERROR] IP columns missing from bin"
            yield pattern_id, ae_id, bin_start, bin_df

    unmatched = df.loc[~df.index.isin(seen_indices)].copy()
    if not unmatched.empty:
        if not pd.api.types.is_datetime64_any_dtype(unmatched['timestamp']):
            temp_ts = pd.to_datetime(unmatched['timestamp'] * 1e9, errors='coerce')
        else:
            temp_ts = unmatched['timestamp']

        unmatched['bin_start'] = temp_ts.dt.floor(f"{bin_size_seconds}s")
        ae_id = hashlib.sha256("unmatched".encode()).hexdigest()[:16]

        for bin_start, bin_df in unmatched.groupby("bin_start"):
            metadata = {
                "pattern_id": "unmatched",
                "ae_id": ae_id,
                "bin_start": bin_start,
                "row_count": len(bin_df),
                "percentage_of_total": len(bin_df) / total_rows
            }
            log_with_tag(logger, logging.INFO, "PatternRouter", f"[Metadata] {metadata}")
            assert 'src_ip' in bin_df.columns and 'dst_ip' in bin_df.columns, "[ERROR] IP columns missing from bin"
            yield "unmatched", ae_id, bin_start, bin_df