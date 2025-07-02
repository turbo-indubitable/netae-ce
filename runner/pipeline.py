import hashlib
import logging
import torch
import pandas as pd
from typing import Dict
from datetime import datetime
import os
from torch.utils.data import TensorDataset

from networkae.db import initialize_db_logger
from networkae.ingest.load_pattern import validate_all_patterns
from networkae.ingest.pattern_router import parse_pattern_filters
from networkae.model.reporting import PostgresLogger
from networkae.ingest.pattern_router import route_and_bin
from networkae.config.config_loader import load_config
from networkae.model.model import NetworkAutoencoder, load_model
from networkae.utils.field_selector import get_field_names
from networkae.utils.transform import preprocess_for_training, preprocess_for_routing
from networkae.utils.timesummary import print_timestamp_summary
from networkae.utils.logging_manager import LoggingManager, log_with_tag
from networkae.model.post_training import log_post_training_summary


def generate_ae_id(pattern_id: str) -> str:
    """Deterministically generate a unique ae_id from a pattern_id."""
    return hashlib.sha256(pattern_id.encode()).hexdigest()[:16]


def run_training_pipeline(
    csv_path: str,
    config_path: str,
    persist_models: bool = True
):
    actual_file_name = os.path.basename(csv_path)
    config = load_config(yaml_path=config_path)
    logger = LoggingManager(config_path=config_path).get_logger(__name__)
    db_logger = initialize_db_logger(config)
    conn = db_logger.conn

    log_with_tag(logger, logging.INFO, "Pipeline", f"Config path: {config_path}")
    speed_mode = config.get("training", {}).get("config_source", {}).get("speed_mode", False)
    log_with_tag(logger, logging.INFO, "Pipeline", f"Speed mode: {'ENABLED' if speed_mode else 'OFF'}")

    if speed_mode:
        raw_df = pd.read_csv(csv_path, nrows=100)
    else:
        raw_df = pd.read_csv(csv_path)

    log_with_tag(logger, logging.INFO, "Pipeline", f"CSV loaded with {len(raw_df)} rows.")
    print_timestamp_summary(raw_df, column="timestamp", logger=logger)

    routing_df = preprocess_for_routing(raw_df)
    log_with_tag(logger, logging.INFO, "Pipeline", "Preprocessing (routing mode) complete.")

    patterns = config.get('routed_traffic_patterns', [])
    if not validate_all_patterns(patterns):
        raise ValueError("One or more routed traffic patterns are invalid.")

    example_df = None
    for _, _, _, candidate_df in route_and_bin(routing_df, parse_pattern_filters(patterns), logger=logger):
        if not candidate_df.empty:
            example_df = candidate_df
            break

    if example_df is None:
        raise RuntimeError("No valid bins found during initial routing — cannot determine field list.")

    processed_df = preprocess_for_training(example_df).drop(columns=[
        'src_ip', 'dst_ip', 'src_port', 'dst_port',
        'packet_size', 'protocol', 'bin_start'
    ], errors='ignore')
    final_base_field_names = get_field_names(processed_df)
    expected_proto_cols = [col for col in final_base_field_names if col.startswith("proto_")]

    reporting = PostgresLogger(db_config=config, logger=logger)
    reporting.insert_routed_patterns_to_db(patterns)
    shared_run_id = reporting.get_run_id()
    ae_map: Dict[str, NetworkAutoencoder] = {}
    bin_samples: Dict[str, pd.DataFrame] = {}
    logged_config = False

    for pattern in patterns:
        pattern_id = pattern['pattern_id']
        if pattern_id == "unmatched":
            log_with_tag(logger, logging.INFO, "Pipeline", "Skipping unmatched model creation")
            continue

        ae_id = generate_ae_id(pattern_id)
        pattern['ae_id'] = ae_id
        model = load_model(
            ae_id=ae_id,
            config=config["training"],
            field_names=final_base_field_names,
            logger=logger,
            file_name=actual_file_name,
            db_logger=reporting,
            speed_mode=speed_mode,
            run_id=shared_run_id
        )
        ae_map[ae_id] = model
        model.db_logger = reporting
        model.run_id = shared_run_id
        if not logged_config:
            reporting.log_training_config(shared_run_id, config)
            logged_config = True

    parsed_patterns = parse_pattern_filters(patterns, logger=logger)

    for pattern_id, ae_id, bin_start, bin_df in route_and_bin(routing_df, parsed_patterns, logger=logger):
        model = ae_map.get(ae_id)
        if pattern_id == "unmatched" or model is None:
            level = logging.INFO if pattern_id == "unmatched" else logging.WARNING
            msg = (f"Skipping unmatched bin at {bin_start} ({len(bin_df)} rows)"
                   if pattern_id == "unmatched"
                   else f"Missing model for ae_id {ae_id} (pattern '{pattern_id}')")
            log_with_tag(logger, level, "Pipeline", msg)
            continue

        bin_df = preprocess_for_training(bin_df, expected_proto_cols=expected_proto_cols).drop(columns=[
            'src_ip', 'dst_ip', 'src_port', 'dst_port',
            'packet_size', 'protocol', 'bin_start'
        ], errors='ignore')

        model.field_names = get_field_names(bin_df)

        if ae_id not in bin_samples:
            bin_samples[ae_id] = bin_df.copy()

        model.train_on_chunk(
            bin_df,
            bin_start=bin_start,
            db_logger=db_logger
        )

    if persist_models:
        for ae_id, model in ae_map.items():
            if model.ae_id == generate_ae_id("unmatched"):
                log_with_tag(logger, logging.INFO, "Pipeline", "Skipping unmatched model post-training")
                continue

            # Assemble new output path: output/{run_id}_{date}/model.pt
            date_str = datetime.utcnow().strftime('%Y-%m-%d')
            output_dir = os.path.join("output", f"{shared_run_id}_{date_str}")
            os.makedirs(output_dir, exist_ok=True)

            # Save each AE model using ae_id
            output_path = os.path.join(output_dir, f"{ae_id}.pt")
            model.save(output_path)

            print(f"[✔] Model {ae_id} saved to: {output_path}")

            log_with_tag(logger, logging.INFO, "Train", f"Model for ae_id {ae_id} saved at {output_path}")

            sample_source_df = bin_samples.get(ae_id)
            if sample_source_df is None or sample_source_df.empty:
                log_with_tag(logger, logging.WARNING, "PostTrain", f"No training sample found for ae_id {ae_id}")
                continue

            sample_df = sample_source_df.sample(n=1, random_state=42)
            processed_sample = preprocess_for_training(sample_df, expected_proto_cols=expected_proto_cols).drop(columns=[
                'src_ip', 'dst_ip', 'src_port', 'dst_port',
                'packet_size', 'protocol', 'bin_start'
            ], errors='ignore')

            if processed_sample.shape[1] != len(model.field_names):
                log_with_tag(logger, logging.ERROR, "PostTrain",
                             f"Mismatch: Sample has {processed_sample.shape[1]} fields, model expects {len(model.field_names)}")
                continue

            dataset = TensorDataset(torch.tensor(processed_sample.values, dtype=torch.float32))
            log_post_training_summary(
                df=sample_df,
                dataset=dataset,
                model=model,
                db_logger=reporting,
                sample_row=sample_df.iloc[0],
                field_names=model.field_names,
                run_id=shared_run_id,
                top_n=10
            )

            if hasattr(model, "field_mse") and hasattr(model, "raw_stats") and hasattr(model, "normalized_stats"):
                reporting.log_field_stats(
                    field_mse=model.field_mse,
                    raw_stats=model.raw_stats,
                    norm_stats=model.normalized_stats,
                    ae_id=ae_id
                )
                log_with_tag(logger, logging.DATABASE, "Train", f"Logged field stats for ae_id={ae_id}")

    conn.close()
    log_with_tag(logger, logging.INFO, "Pipeline", "Training complete.")