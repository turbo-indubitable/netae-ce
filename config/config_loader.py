import yaml
import os
import psutil
import torch
import logging
import pprint

from networkae.config.config_validator import validate_config
from networkae.config.global_config import GlobalConfig


DEFAULT_EPOCHS = 50
DEFAULT_MODEL_MB = 50
DEFAULT_BATCH_CAP = 8192
DEFAULT_MIN_BATCH = 32


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


def load_yaml(yaml_path, logger=None):
    if not os.path.exists(yaml_path):
        log_with_tag(logger, logging.WARNING, "ConfigLoad", f"YAML path not found: {yaml_path}")
        return {}
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        log_with_tag(logger, logging.ERROR, "ConfigLoad", f"Failed to load YAML from {yaml_path}: {e}")
        return {}


def auto_detect_training_settings(model_size_mb=DEFAULT_MODEL_MB, safety_margin_pct=10):
    total_memory = psutil.virtual_memory().available

    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory

    usable_memory = total_memory * (1 - safety_margin_pct / 100.0)
    est_batch = int(usable_memory / (model_size_mb * 1024 ** 2))
    batch_size = max(DEFAULT_MIN_BATCH, min(est_batch, DEFAULT_BATCH_CAP))

    return {
        "num_epochs": DEFAULT_EPOCHS,
        "batch_size": batch_size,
        "safety_margin_pct": safety_margin_pct
    }


def load_config(yaml_path=None, logger=None) -> GlobalConfig:
    logger = logger or NullLogger()

    raw_config = load_yaml(yaml_path, logger=logger) if yaml_path else {}
    validate_config(raw_config)

    log_with_tag(logger, logging.DEBUG, "ConfigLoad", f"YAML config loaded successfully.")
    log_with_tag(logger, logging.DEBUG, "ConfigLoad", "Full validated config:")
    log_with_tag(logger, logging.DEBUG, "ConfigLoad", pprint.pformat(raw_config))

    return GlobalConfig(raw_config)