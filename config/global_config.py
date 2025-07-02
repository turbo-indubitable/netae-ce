class GlobalConfig:
    def __init__(self, config_dict: dict):
        self.raw = config_dict
        self.general = config_dict.get("General", {})
        self.config_source = config_dict.get("config_source", {})
        self.run_parameters = config_dict.get("run_parameters", {})
        self.training = config_dict.get("training", {})
        self.early_stopping = config_dict.get("early_stopping", {})
        self.routed_traffic_patterns = config_dict.get("routed_traffic_patterns", [])
        self.db = config_dict.get("db", {})

    def get(self, key, default=None):
        return self.raw.get(key, default)