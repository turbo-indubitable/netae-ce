import logging
import traceback
from typing import Optional

import pandas as pd

from networkae.pipeline.activate import activate_pipeline
from networkae.pipeline.data_process import process_routing_and_sample_fields, generate_binned_traffic
from networkae.pipeline.train import create_model_map, train_on_binned_data
from networkae.pipeline.document import persist_models_and_log_summaries
from networkae.model.reporting import PostgresLogger
from networkae.utils.logging_manager import LoggingManager, log_with_tag
from networkae.ingest.pattern_router import parse_pattern_filters
from networkae.config.global_config import GlobalConfig
from networkae.utils.statistics import compute_stats_with_protocol


class PipelineRunner:
    def __init__(self, csv_path: str, config_path: str, persist_models: bool = True):
        self.csv_path = csv_path
        self.config_path = config_path
        self.persist_models = persist_models

        self.logger = LoggingManager(config_path=config_path).get_logger("Pipeline")

        # Initialize placeholders
        self.config: Optional[GlobalConfig] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.actual_file_name: Optional[str] = None
        self.db_logger: Optional[PostgresLogger] = None
        self.patterns = []
        self.parsed_patterns = []
        self.routing_df = None
        self.expected_proto_cols = []
        self.field_names = []
        self.binned_data = []
        self.run_id = None
        self.ae_map = None

    def run(self) -> None:
        """Main entry point for pipeline execution."""
        try:
            self._log_info("Starting pipeline execution.")
            self._activate()
            self._preprocess_and_bin()
            self._initialize_models()
            bin_samples = self._train_models()

            if self.persist_models:
                self._persist_models(bin_samples)

            self._log_info("Pipeline execution complete.")
            if self.db_logger:
                self.db_logger.close()

        except Exception as e:
            self._log_error(f"Unhandled exception: {str(e)}")
            self._log_debug(traceback.format_exc())
            raise

    def _activate(self) -> None:
        """Initializes configuration, logging, database connection, and loads the input CSV."""
        self._log_info("Initializing config, DB, and input data.")

        self.config, _, self.db_logger, self.raw_df, self.actual_file_name = activate_pipeline(
            csv_path=self.csv_path,
            config_path=self.config_path
        )

        self._log_debug("========== Config Summary ==========")
        self._log_debug(f"Training epochs: {self.config.training.get('num_epochs')}")
        self._log_debug(f"Batch size: {self.config.training.get('batch_size')}")
        self._log_debug(f"Speed mode: {self.config.run_parameters.get('speed_mode')}")
        self._log_debug(f"Learning rate decay factor: {self.config.training.get('lr_decay_factor')}")
        self._log_debug(f"Patterns loaded: {len(self.patterns)}")
        self._log_debug("====================================")

        self.patterns = self.config.routed_traffic_patterns
        self.parsed_patterns = parse_pattern_filters(self.patterns, logger=self.logger)

    def _preprocess_and_bin(self) -> None:
        """Preprocesses the raw DataFrame and generates binned segments per traffic pattern."""
        self._log_info("Beginning data preprocessing and field extraction.")

        self.routing_df, full_field_names, self.expected_proto_cols = process_routing_and_sample_fields(
            self.raw_df, self.config, self.logger, self.parsed_patterns
        )

        self._log_info("Routing and binning data once for training.")
        self.binned_data = generate_binned_traffic(
            routing_df=self.routing_df,
            parsed_patterns=self.parsed_patterns,
            logger=self.logger
        )

        sample_df = next((df for _, _, _, df in self.binned_data if not df.empty), None)
        if sample_df is None:
            raise ValueError("No non-empty bins found — cannot determine model input shape.")

        fields_to_drop = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'packet_size', 'protocol', 'bin_start']
        self.field_names = [col for col in sample_df.columns if col not in fields_to_drop]

        self._log_debug(f"Final field count going to model: {len(self.field_names)}")
        self._log_debug(f"Fields used by model: {self.field_names}")

        # 🧠 Log protocol distribution (once per run)
        raw_stats = compute_stats_with_protocol(self.routing_df)
        if "protocol_distribution" in raw_stats and self.db_logger:
            self.db_logger.log_protocol_distribution(
                stat_type="raw",
                stats_dict={"protocol_distribution": raw_stats["protocol_distribution"]},
                run_id=self.run_id
            )

    def _initialize_models(self) -> None:
        """Initializes the database logger and autoencoder model instances per pattern."""
        self._log_info("Setting up DB logging and AE model map.")
        self.db_logger.insert_routed_patterns_to_db(self.patterns)
        self.run_id = self.db_logger.get_run_id()
        speed_mode = self.config.run_parameters.get("speed_mode", False)

        self.ae_map = create_model_map(
            patterns=self.patterns,
            field_names=self.field_names,
            actual_file_name=self.actual_file_name,
            config=self.config,
            logger=self.logger,
            reporting=self.db_logger,
            speed_mode=speed_mode,
            run_id=self.run_id
        )

        self._log_debug(f"Speed mode: {speed_mode} (from config.run_parameters.speed_mode)")

    def _train_models(self):
        """Trains each model using the binned data."""
        self._log_info("Starting AE training over preprocessed bins.")
        return train_on_binned_data(
            binned_data=self.binned_data,
            ae_map=self.ae_map,
            expected_proto_cols=self.expected_proto_cols,
            db_logger=self.db_logger,
            logger=self.logger
        )

    def _persist_models(self, bin_samples):
        """Saves model weights and logs training summary metadata."""
        self._log_info("Saving models and logging final summaries.")
        persist_models_and_log_summaries(
            ae_map=self.ae_map,
            bin_samples=bin_samples,
            expected_proto_cols=self.expected_proto_cols,
            run_id=self.run_id,
            config=self.config,
            logger=self.logger,
            persist_models=True
        )

    def _log_info(self, msg: str) -> None:
        log_with_tag(self.logger, logging.INFO, "Pipeline", msg)

    def _log_debug(self, msg: str) -> None:
        log_with_tag(self.logger, logging.DEBUG, "Pipeline", msg)

    def _log_error(self, msg: str) -> None:
        log_with_tag(self.logger, logging.ERROR, "Pipeline", msg)