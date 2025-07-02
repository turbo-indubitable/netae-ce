# networkae/entrypoints/run_single.py

import argparse
import os
import torch
from datetime import datetime

from networkae.model.model import NetworkAutoencoder
from networkae.ingest.ingest import load_data, create_dataset
from networkae.model.reporting import PostgresLogger
from networkae.utils.logging_manager import LoggingManager


def main(config):
    log_mgr = LoggingManager(config_path=config.get("logging_config"))
    logger = log_mgr.get_logger("run_single")

    file_path = config["input_path"]
    logger.info(f"Loading and preprocessing: {file_path}")

    df, raw_stats, norm_stats = load_data(file_path)
    dataset, sample_row = create_dataset(df)

    input_dim = df.shape[1]
    file_name = os.path.basename(file_path)
    field_names = df.columns.tolist()

    logger.info(f"Initializing model for input_dim={input_dim}")

    db_logger = PostgresLogger(
        dbname=config["db"].get("name"),
        user=config["db"].get("user"),
        password=config["db"].get("password"),
        host=config["db"].get("host", "localhost"),
        port=config["db"].get("port", 5432)
    )

    model = NetworkAutoencoder(
        input_dim=input_dim,
        config=config,
        db_logger=db_logger,
        file_name=file_name,
        field_names=field_names
    )

    db_logger.log_field_stats([0.0] * input_dim, raw_stats, norm_stats)  # Initial logging
    model.train(dataset)

    sample_tensor = torch.tensor(sample_row.values, dtype=torch.float32).unsqueeze(0)
    mse, cosine_sim = model.evaluate(sample_tensor)
    db_logger.update_run_metrics(reconstructed_mse=mse, reconstructed_cosine=cosine_sim)

    logger.info(f"Finished: {file_path}, MSE={mse:.6f}, Cosine={cosine_sim:.6f}")
    db_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--log", default="config/logging.yaml", help="Logger config path")
    args = parser.parse_args()

    import yaml

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    base_config["input_path"] = args.input
    base_config["logging_config"] = args.log

    main(base_config)
