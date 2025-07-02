# networkae/utils/netsim_logging_manager.py
import sys, yaml, logging, psycopg2
import logging.handlers
from pathlib import Path

# === Log Level Reference Table ===
# | Level Name | Numeric Value | Purpose                             |
# |------------|----------------|-------------------------------------|
# | CRITICAL   | 50             | Fatal errors, system shutdown       |
# | ERROR      | 40             | Serious issues, recoverable errors  |
# | WARNING    | 30             | Something unexpected, still running |
# | SUMMARY    | 25             | High-level overview                 |
# | INFO       | 20             | General information                 |
# | DEBUG      | 10             | Developer debug messages            |
# | TRAIN      |  7             | Training progress logs              |
# | DATABASE   |  5             | DB operations, low-level details    | (insanely verbose, careful if storing)



# Python only considers messages with level >= current logger level.
# So setting log level to SUMMARY (45) will show: SUMMARY, WARNING, ERROR, CRITICAL.


# === Custom Log Levels ===
SUMMARY_LEVEL_NUM = 25
DATABASE_LEVEL_NUM = 5
TRAIN_LEVEL_NUM = 7

logging.addLevelName(SUMMARY_LEVEL_NUM, "SUMMARY")
logging.addLevelName(DATABASE_LEVEL_NUM, "DATABASE")
logging.addLevelName(TRAIN_LEVEL_NUM, "TRAIN")

def summary(self, message, *args, **kwargs):
    if self.isEnabledFor(SUMMARY_LEVEL_NUM):
        self._log(SUMMARY_LEVEL_NUM, message, args, **kwargs)

def database(self, message, *args, **kwargs):
    if self.isEnabledFor(DATABASE_LEVEL_NUM):
        self._log(DATABASE_LEVEL_NUM, message, args, **kwargs)

def train(self, message, *args, **kwargs):
    if self.isEnabledFor(TRAIN_LEVEL_NUM):
        self._log(TRAIN_LEVEL_NUM, message, args, **kwargs)

logging.Logger.summary = summary
logging.Logger.database = database
logging.Logger.train = train
logging.SUMMARY = SUMMARY_LEVEL_NUM
logging.DATABASE = DATABASE_LEVEL_NUM
logging.TRAIN = TRAIN_LEVEL_NUM

# === Color codes ===
LOG_COLORS = {
    "SUMMARY": "\033[1;38;5;245m",      # Bold + grayscale 245 (medium gray)
    "DEBUG": "\033[94m",                # Blue
    "TRAIN": "\033[38;5;208m",          # Orange
    "DATABASE": "\033[96m",             # Cyan
    "INFO": "\033[92m",                 # Green
    "WARNING": "\033[93m",              # Yellow
    "ERROR": "\033[91m",                # Red
    "CRITICAL": "\033[95m",             # Magenta
    "RESET": "\033[0m"
}

class TagInjectingFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'tag'):
            record.tag = "General"
        return True

class ColorFormatter(logging.Formatter):
    def format(self, record):
        base_level = record.levelname.replace(LOG_COLORS["RESET"], "").strip()
        color = LOG_COLORS.get(base_level, "")
        reset = LOG_COLORS["RESET"]

        record.levelname = f"{color}{record.levelname}{reset}"
        record.name = f"{color}{record.name}{reset}"
        if hasattr(record, "tag"):
            record.tag = f"{color}{record.tag}{reset}"
        record.msg = f"{color}{record.msg}{reset}"

        return super().format(record)

class LoggingManager:
    def __init__(self, config_path=None, use_db=False, db_config=None):
        self.logger = None
        self.config = {
            "level": "INFO",
            "file": "app.log",
            "max_bytes": 10 * 1024 * 1024,
            "backup_count": 5,
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(tag)s] %(message)s",
            "console": True,
            "console_level": "DEBUG"
        }

        if use_db and db_config:
            self._load_config_from_db(db_config)
        elif config_path:
            self._load_config_from_yaml(config_path)

        self._setup_logger()

    def _load_config_from_yaml(self, path):
        try:
            with open(path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                general_config = yaml_config.get("General", {})
                self.config.update(general_config)
        except Exception as e:
            print(f"Failed to load YAML config: {e}")

    def _load_config_from_db(self, db_config):
        try:
            conn = psycopg2.connect(**db_config)
            with conn.cursor() as cur:
                cur.execute("SELECT key, value FROM logging_config")
                for key, value in cur.fetchall():
                    if key in ["max_bytes", "backup_count"]:
                        value = int(value)
                    if key == "console":
                        value = value.lower() == "true"
                    self.config[key] = value
            conn.close()
        except Exception as e:
            print(f"Failed to load DB config: {e}")

    def _setup_logger(self):
        log_level = getattr(logging, self.config["level"].upper(), logging.INFO)
        console_level = getattr(logging, self.config["console_level"].upper(), logging.DEBUG)

        logger_name = self.config.get("logger_name", "Pipeline")
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Don't attach handlers again if they already exist
        if self.logger.handlers:
            return

        formatter = logging.Formatter(self.config["format"])
        color_formatter = ColorFormatter(self.config["format"])

        file_handler = logging.handlers.RotatingFileHandler(
            self.config["file"],
            maxBytes=self.config["max_bytes"],
            backupCount=self.config["backup_count"]
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        file_handler.addFilter(TagInjectingFilter())
        self.logger.addHandler(file_handler)

        if self.config.get("console"):
            console_handler = logging.StreamHandler(stream=sys.stdout)
            console_handler.setFormatter(color_formatter)
            console_handler.setLevel(console_level)
            console_handler.addFilter(TagInjectingFilter())
            self.logger.addHandler(console_handler)

        print("[LoggingManager] Logger initialized")
        print(f"    → Logger name: {logger_name}")
        print(f"    → Configured log level: {self.config['level']}")
        print(f"    → Console log level: {self.config['console_level']}")
        print(f"    → File log path: {self.config['file']}")
        print(
            f"    → Custom levels: SUMMARY={SUMMARY_LEVEL_NUM}, DATABASE={DATABASE_LEVEL_NUM}, TRAIN={TRAIN_LEVEL_NUM}")

    def get_logger(self, name_or_module="Pipeline"):
        return self.logger  # already named + initialized

def log_with_tag(logger, level, tag, message):
    logger.log(level, message, extra={"tag": tag})