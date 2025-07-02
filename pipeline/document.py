# networkae/pipeline/document.py

import os
import logging
from typing import Dict, List

import torch
import pandas as pd
from torch.utils.data import TensorDataset

from networkae.model.model import NetworkAutoencoder
from networkae.pipeline.data_process import generate_ae_id
from networkae.utils.transform import preprocess_for_training_sample
from networkae.utils.field_selector import get_field_names
from networkae.utils.logging_manager import log_with_tag
from networkae.model.post_training import log_post_training_summary
from networkae.config.global_config import GlobalConfig


def persist_models_and_log_summaries(
    ae_map: Dict[str, NetworkAutoencoder],
    bin_samples: Dict[str, pd.DataFrame],
    expected_proto_cols: List[str],
    run_id: str,
    config: GlobalConfig,
    logger,
    persist_models: bool = True
) -> None:
    """
    Saves models, logs sample-based summaries, and records per-field stats.
    Assumes bin_samples were already routed and preprocessed.

    Args:
        ae_map: Mapping from ae_id to trained model
        bin_samples: One representative sample DataFrame per ae_id
        expected_proto_cols: One-hot encoded protocol columns
        run_id: Unique run ID
        config: Global configuration object
        logger: Logger instance
        persist_models: If True, saves models to disk
    """
    output_dir = config.training.get("output_dir", "./output/")
    os.makedirs(output_dir, exist_ok=True)

    for ae_id, model in ae_map.items():
        if model.ae_id == generate_ae_id("unmatched"):
            log_with_tag(logger, logging.INFO, "PostTrain", "Skipping unmatched model post-training")
            continue

        if persist_models:
            model_path = os.path.join(output_dir, f"{ae_id}.pt")
            model.save(model_path)
            log_with_tag(logger, logging.INFO, "PostTrain", f"Model saved at {model_path}")

        sample_df = bin_samples.get(ae_id)
        if sample_df is None or sample_df.empty:
            log_with_tag(logger, logging.WARNING, "PostTrain", f"No sample found for ae_id {ae_id}")
            continue

        row = sample_df.sample(n=1, random_state=42)

        processed = preprocess_for_training_sample(
            row, expected_proto_cols=expected_proto_cols
        )

        processed = processed.drop(columns=[
            'src_ip', 'dst_ip', 'src_port', 'dst_port',
            'packet_size', 'protocol', 'bin_start'
        ], errors='ignore')

        if processed.shape[1] != len(model.field_names):
            log_with_tag(logger, logging.ERROR, "PostTrain",
                         f"Mismatch: Sample has {processed.shape[1]} fields, model expects {len(model.field_names)}")
            continue

        dataset = TensorDataset(torch.tensor(processed.values, dtype=torch.float32))

        log_with_tag(logger, logging.DEBUG, "PostTrain", f"Field MSE present: {hasattr(model, 'field_mse')}")
        log_with_tag(logger, logging.DEBUG, "PostTrain", f"Raw stats present: {hasattr(model, 'raw_stats')}")
        log_with_tag(logger, logging.DEBUG, "PostTrain", f"Norm stats present: {hasattr(model, 'normalized_stats')}")

        log_post_training_summary(
            df=row,
            dataset=dataset,
            model=model,
            db_logger=model.db_logger,
            sample_row=row.iloc[0],
            field_names=model.field_names,
            run_id=run_id,
            top_n=10
        )

        if hasattr(model, "field_mse") and hasattr(model, "raw_stats") and hasattr(model, "normalized_stats"):
            model.db_logger.log_field_stats(
                run_id=model.run_id,
                ae_id=model.ae_id,
                field_mse=model.field_mse,
                raw_stats=model.raw_stats,
                norm_stats=model.normalized_stats,
                field_names=model.field_names
            )
            log_with_tag(logger, logging.DATABASE, "PostTrain", f"Logged field stats for ae_id={ae_id}")