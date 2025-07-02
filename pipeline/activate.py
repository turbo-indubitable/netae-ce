# networkae/pipeline/activate.py

import os
import logging
import pandas as pd
from typing import Tuple

from networkae.config.config_loader import load_config
from networkae.config.global_config import GlobalConfig
from networkae.db import initialize_db_logger
from networkae.utils.logging_manager import LoggingManager, log_with_tag
from networkae.utils.timesummary import print_timestamp_summary


def activate_pipeline(
    csv_path: str,
    config_path: str
) -> Tuple[GlobalConfig, logging.Logger, object, pd.DataFrame, str]:
    """
    Activation stage: load config, logger, DB, CSV, and timestamp summary.

    Returns:
        config (GlobalConfig): Structured and validated config object
        logger (logging.Logger): Application logger
        db_logger (PostgresLogger): Initialized DB logger
        raw_df (pd.DataFrame): Raw DataFrame (limited if speed mode is on)
        actual_file_name (str): Base filename of the CSV
    """
    config = load_config(yaml_path=config_path)
    logger = LoggingManager(config_path=config_path).get_logger("Pipeline")
    db_logger = initialize_db_logger(config, logger=logger)

    log_with_tag(logger, logging.INFO, "Activate", f"Config path: {config_path}")

    speed_mode = config.run_parameters.get("speed_mode", False)
    log_with_tag(logger, logging.INFO, "Activate", f"Speed mode: {'ENABLED' if speed_mode else 'OFF'}")

    # Load CSV (limited if speed mode)
    if speed_mode:
        raw_df = pd.read_csv(csv_path, nrows=100)
    else:
        raw_df = pd.read_csv(csv_path)

    actual_file_name = os.path.basename(csv_path)
    log_with_tag(logger, logging.INFO, "Activate", f"CSV loaded with {len(raw_df):,} rows.")

    # Summarize timestamp range
    print_timestamp_summary(raw_df, column="timestamp", logger=logger)

    return config, logger, db_logger, raw_df, actual_file_name