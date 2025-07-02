import yaml

EXPECTED_SCHEMA = {
    "General": {
        "level": str,
        "file": str,
        "max_bytes": int,
        "backup_count": int,
        "console": bool,
        "console_level": str,
    },
    "config_source": {
        "prefer": str,
        "fallback": str
    },
    "run_parameters": {
        "seed": (int, type(None)),
        "speed_mode": bool
    },
    "training": {
        "num_epochs": (int, type(None)),
        "batch_size": (int, type(None)),
        "min_batches_per_bin": int,
        "max_batches_per_bin": int,
        "use_hw_autodetect": bool,
        "lr_decay_factor": float,
        "lr_decay_patience": int,
        "safety_margin_pct": int
    },
    "early_stopping": {
        "patience": int,
        "min_delta": float,
        "floor": float
    },
    "db": {
        "host": str,
        "dbname": str,
        "user": str,
        "password": str,
        "port": int
    },
    "routed_traffic_patterns": list
}

def coerce_type(value, expected_type):
    if isinstance(expected_type, tuple):
        for t in expected_type:
            try:
                return t(value) if value is not None else None
            except (ValueError, TypeError):
                continue
        raise ValueError(f"Cannot coerce {value} to any of {expected_type}")
    return expected_type(value)

def validate_section(config_section, schema_section, path=""):
    validated = {}
    for key, expected_type in schema_section.items():
        full_key = f"{path}.{key}" if path else key
        if key not in config_section:
            raise ValueError(f"Missing config key: {full_key}")

        value = config_section[key]

        if isinstance(expected_type, dict):
            if not isinstance(value, dict):
                raise TypeError(f"Expected dict for {full_key}, got {type(value).__name__}")
            validated[key] = validate_section(value, expected_type, full_key)
        elif expected_type == list:
            if not isinstance(value, list):
                raise TypeError(f"Expected list for {full_key}, got {type(value).__name__}")
            validated[key] = value
        else:
            try:
                validated[key] = coerce_type(value, expected_type)
            except Exception as e:
                raise TypeError(f"Failed to coerce {full_key}: {e}")
    return validated


def validate_config(config):
    validated_config = {}
    for section_key, section_schema in EXPECTED_SCHEMA.items():
        if section_key not in config:
            raise ValueError(f"Missing top-level section: {section_key}")

        value = config[section_key]

        if section_schema == list:
            if not isinstance(value, list):
                raise TypeError(f"Expected list for section {section_key}, got {type(value).__name__}")
            validated_config[section_key] = value

        elif isinstance(section_schema, dict):
            validated_config[section_key] = validate_section(value, section_schema, section_key)

        else:
            try:
                validated_config[section_key] = coerce_type(value, section_schema)
            except Exception as e:
                raise TypeError(f"Failed to coerce {section_key}: {e}")

    return validated_config

def load_and_validate_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return validate_config(config)