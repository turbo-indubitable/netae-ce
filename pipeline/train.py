import os
import logging
import pandas as pd
from typing import Dict, List, Tuple

from networkae.model.model import NetworkAutoencoder, load_model
from networkae.utils.field_selector import get_field_names
from networkae.utils.logging_manager import log_with_tag
from networkae.pipeline.data_process import generate_ae_id
from networkae.config.global_config import GlobalConfig


def create_model_map(
    patterns: list,
    field_names: list,
    actual_file_name: str,
    config: GlobalConfig,
    logger,
    reporting,
    speed_mode: bool,
    run_id: str
) -> Dict[str, NetworkAutoencoder]:
    """Load or initialize one AE model per pattern."""
    ae_map = {}
    logged_config = False

    for pattern in patterns:
        pattern_id = pattern["pattern_id"]
        if pattern_id == "unmatched":
            log_with_tag(logger, logging.INFO, "Train", "Skipping unmatched model creation")
            continue

        ae_id = generate_ae_id(pattern_id)
        pattern["ae_id"] = ae_id  # attach for reuse

        model = load_model(
            ae_id=ae_id,
            training_config=config.training,
            early_stopping_config=config.early_stopping,
            field_names=field_names,
            logger=logger,
            file_name=actual_file_name,
            db_logger=reporting,
            speed_mode=speed_mode,
            run_id=run_id
        )
        model.db_logger = reporting
        model.run_id = run_id

        ae_map[ae_id] = model

        if not logged_config:
            reporting.log_training_config(run_id, config.raw)  # Log the full original config
            logged_config = True

    log_with_tag(logger, logging.DEBUG, "Train", f"Run ID: {run_id} (auto-assigned)")
    log_with_tag(logger, logging.DEBUG, "Train", f"Model file: {actual_file_name}")
    log_with_tag(logger, logging.DEBUG, "Train", f"Speed mode: {speed_mode} (from config.run_parameters.speed_mode)")
    return ae_map


def train_on_binned_data(
    binned_data: List[Tuple[str, str, str, pd.DataFrame]],
    ae_map: Dict[str, NetworkAutoencoder],
    expected_proto_cols: list,
    db_logger,
    logger
) -> Dict[str, pd.DataFrame]:
    """
    Train each AE model on its corresponding bins.

    Args:
        binned_data: List of (pattern_id, ae_id, bin_start, bin_df)
        ae_map: Dict of AE instances by ae_id
        expected_proto_cols: List of protocol one-hot columns to preserve
        db_logger: Database logger
        logger: Logger instance

    Returns:
        Dict of one sample bin per AE for post-training analysis
    """
    bin_samples = {}

    for pattern_id, ae_id, bin_start, bin_df in binned_data:
        model = ae_map.get(ae_id)
        if pattern_id == "unmatched" or model is None:
            level = logging.INFO if pattern_id == "unmatched" else logging.WARNING
            msg = (f"Skipping unmatched bin at {bin_start} ({len(bin_df)} rows)"
                   if pattern_id == "unmatched"
                   else f"Missing model for ae_id {ae_id} (pattern '{pattern_id}')")
            log_with_tag(logger, level, "Train", msg)
            continue

        # Drop any routing-only columns that shouldn't go into training
        bin_df = bin_df.drop(columns=[
            'src_ip', 'dst_ip', 'src_port', 'dst_port',
            'packet_size', 'protocol', 'bin_start'
        ], errors='ignore')

        # Ensure model gets field names dynamically
        model.field_names = get_field_names(bin_df)

        # Save one sample bin for later
        if ae_id not in bin_samples:
            bin_samples[ae_id] = bin_df.copy()

        model.train_on_chunk(
            bin_df,
            bin_start=bin_start,
            db_logger=db_logger
        )

    return bin_samples