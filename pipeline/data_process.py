# networkae/pipeline/data_process.py

import logging, hashlib
import pandas as pd
from typing import Tuple, List

from networkae.utils.transform import preprocess_for_routing, preprocess_for_training
from networkae.utils.field_selector import get_field_names
from networkae.ingest.pattern_router import route_and_bin, parse_pattern_filters
from networkae.ingest.load_pattern import validate_all_patterns
from networkae.utils.logging_manager import log_with_tag


def generate_binned_traffic(
    routing_df: pd.DataFrame,
    parsed_patterns: list,
    logger
) -> list:
    """
    Runs route_and_bin once and returns a list of matched bins.
    Each entry is (pattern_id, ae_id, bin_start, bin_df).
    """
    binned_output = list(route_and_bin(routing_df, parsed_patterns, logger=logger))
    log_with_tag(logger, logging.INFO, "DataProcess", f"Generated {len(binned_output)} binned time segments.")
    return binned_output

def extract_valid_sample_df(
    routing_df: pd.DataFrame,
    parsed_patterns: list,
    logger
) -> pd.DataFrame:
    """
    Finds the first non-empty candidate_df after routing. Used to extract fields for training.
    Logs are suppressed unless explicitly requested.
    """
    for pattern_id, ae_id, bin_start, candidate_df in route_and_bin(
        routing_df, parsed_patterns, logger=logger):
        if not candidate_df.empty:
            log_with_tag(logger, logging.DEBUG, "DataProcess", f"Selected sample from pattern: {pattern_id}")
            return candidate_df

    raise RuntimeError("No valid bins found during initial routing — cannot determine field list.")


def generate_ae_id(pattern_id: str) -> str:
    """Deterministically generate a unique ae_id from a pattern_id."""
    return hashlib.sha256(pattern_id.encode()).hexdigest()[:16]

def process_routing_and_sample_fields(
    df: pd.DataFrame,
    config: dict,
    logger,
    parsed_patterns: list
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Handles early-stage preprocessing: routing, binning, and field name extraction.

    Returns:
        routing_df (pd.DataFrame): minimally preprocessed data for routing
        field_names (List[str]): column names used for AE training input
        expected_proto_cols (List[str]): proto_* fields that should be preserved
    """
    log_with_tag(logger, logging.INFO, "DataProcess", "Starting routing preprocessing.")
    routing_df = preprocess_for_routing(df.copy())
    log_with_tag(logger, logging.INFO, "DataProcess", "Preprocessing (routing mode) complete.")

    # Validate pattern syntax
    patterns = config.get('routed_traffic_patterns', [])
    if not validate_all_patterns(patterns):
        raise ValueError("One or more routed traffic patterns are invalid.")

    example_df = extract_valid_sample_df(routing_df, parsed_patterns, logger)

    if example_df is None:
        log_with_tag(logger, logging.ERROR, "DataProcess",
                     "No valid bins found during initial routing — cannot determine field list")
        raise RuntimeError("No valid bins found during initial routing — cannot determine field list.")

    # Full training preprocessing on sample
    processed_df = preprocess_for_training(example_df).drop(columns=[
        'src_ip', 'dst_ip', 'src_port', 'dst_port',
        'packet_size', 'protocol', 'bin_start'
    ], errors='ignore')

    # Extract field names and proto columns
    field_names = get_field_names(processed_df)
    expected_proto_cols = [col for col in field_names if col.startswith("proto_")]

    return routing_df, field_names, expected_proto_cols