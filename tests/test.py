# test_driver.py
import os
import pprint
import numpy as np
import random
import torch


from networkae.config.config_loader import load_config
from networkae.db import initialize_db_logger
from networkae.model.post_training import log_post_training_summary
from networkae.model.model import NetworkAutoencoder
from ingest.ingest import load_data, create_dataset
from networkae.utils.logging_manager import LoggingManager

# ---- Configuration ----
CONFIG_PATH = "../config/config.yaml"
LOG_CONFIG_PATH = "../config/logging.yaml"
TEST_FILE = "../../data/testdata.csv"

# ---- Load config from YAML + DB + autodetect fallback ----
config = load_config(yaml_path=CONFIG_PATH, db_conn=None)  # Replace with your DB conn if ready
training_config = config["training"]
config["input_path"] = TEST_FILE
config["logging_config"] = LOG_CONFIG_PATH

# ---- Set up logging ----
log_mgr = LoggingManager(config_path=LOG_CONFIG_PATH)
logger = log_mgr.get_logger("test_driver")

seed = training_config.get("seed")
if seed is not None:
    logger.info(f"Setting random seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- Load + preprocess ----
logger.info(f"Loading test data from {TEST_FILE}")
df, raw_stats, norm_stats = load_data(TEST_FILE)
dataset, sample_row = create_dataset(df)
input_dim = df.shape[1]
field_names = df.columns.tolist()

# ---- Setup DB Logger ----
db_logger = initialize_db_logger(config)

# ---- Log resolved config ----
logger.info("Final resolved training configuration:")
logger.info(pprint.pformat(training_config, indent=2))
print("=== Final Resolved Training Config ===")
pprint.pprint(training_config, indent=2)
print("======================================")

# ---- trim result set if speed mode is enabled ----
if training_config.get("speed_mode", False):
    logger.warning("[SPEED MODE] Only the top 100 rows of data will be used!")
    print(f"[RUN] Speed Mode: {'ENABLED' if training_config.get('speed_mode') else 'OFF'}")
    df = df.head(100)

    if db_logger and hasattr(db_logger, "mark_speed_mode"):
        db_logger.mark_speed_mode()

# ---- Initialize model ----
model = NetworkAutoencoder(
    input_dim=input_dim,
    config=training_config,  # Just pass the training section
    db_logger=db_logger,
    file_name=os.path.basename(TEST_FILE),
    field_names=field_names
)

# Logging input data metrics
db_logger.log_field_stats([0.0] * input_dim, raw_stats, norm_stats)
db_logger.log_protocol_distribution("normalized", norm_stats)
db_logger.log_protocol_distribution("raw", raw_stats)

# ---- Train model ----
model.train(dataset)

# log the config parameters with the run
if model.run_id:
    db_logger.log_training_config(model.run_id, training_config)

# ---- Post training summary ----
log_post_training_summary(
    df=df,
    dataset=dataset,
    model=model,
    db_logger=db_logger,
    sample_row=sample_row,
    field_names=field_names,
    top_n=10
)

db_logger.close()