import uuid
import json
import time
import psycopg2
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

from networkae.utils.logging_manager import log_with_tag


class PostgresLogger:
    def __init__(self, db_config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        host = db_config.get("host")
        dbname = db_config.get("dbname")
        user = db_config.get("user")
        password = db_config.get("password")
        port = db_config.get("port")

        self.run_id: str = str(uuid.uuid4())
        self.conn = psycopg2.connect(
            host=host, dbname=dbname, user=user, password=password, port=port
        )
        self.cursor = self.conn.cursor()
        self.logger = logger

        if self.logger:
            log_with_tag(self.logger, logging.INFO, "PostgresLogger",
                         f"[Init] Connected to PostgreSQL DB '{dbname}' on host '{host}' as user '{user}'")

    @staticmethod
    def generate_run_id() -> str:
        return str(uuid.uuid4())

    def get_run_id(self) -> str:
        return self.run_id

    def log_bin_stats(self, ae_id: str, bin_start: str, mse: float,
                      cosine: Optional[float] = None, count: int = 0) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ae_bin_stats (ae_id, bin_start, mse, cosine_similarity, sample_count)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (ae_id, bin_start) DO UPDATE
                  SET mse = EXCLUDED.mse,
                      cosine_similarity = EXCLUDED.cosine_similarity,
                      sample_count = EXCLUDED.sample_count;
                """,
                (ae_id, bin_start, mse, cosine, count)
            )
        self.conn.commit()

    def load_patterns_from_db(self) -> List[Dict[str, str]]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT pattern_id, filter_expr FROM known_patterns ORDER BY id ASC")
            rows = cur.fetchall()
        return [{'pattern_id': row[0], 'filter_expr': row[1]} for row in rows]

    def insert_routed_patterns_to_db(self, patterns: List[Dict[str, str]]) -> None:
        with self.conn.cursor() as cur:
            for i, pattern in enumerate(patterns):
                pattern_id = pattern['pattern_id']
                filter_expr = pattern['filter_expr']
                filter_order = pattern.get('filter_order', i)

                cur.execute(
                    """
                    INSERT INTO routed_traffic_patterns (pattern_id, filter_expr, filter_order)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (pattern_id) DO UPDATE
                      SET filter_expr = EXCLUDED.filter_expr,
                          filter_order = EXCLUDED.filter_order;
                    """,
                    (pattern_id, filter_expr, filter_order)
                )
        self.conn.commit()

    def log_training_config(self, run_id: str, config: Dict[str, Any]) -> None:
        if not run_id or not config:
            return

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO run_configs (run_id, config)
                VALUES (%s, %s)
                ON CONFLICT (run_id) DO UPDATE SET config = EXCLUDED.config;
                """,
                (run_id, json.dumps(config))
            )
        self.conn.commit()

    def start_run(self, file_name: str, input_dim: int, final_loss: float, best_epoch: int,
                  mse: float, model_arch: Dict[str, Any], speed_mode: bool = False) -> str:
        self.cursor.execute("""
            INSERT INTO runs (
                run_id, file_name, timestamp, input_dim,
                final_loss, best_epoch, mse, model_arch, speed_mode
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            self.run_id,
            file_name,
            datetime.now(),
            input_dim,
            final_loss,
            best_epoch,
            mse,
            json.dumps(model_arch),
            speed_mode
        ))
        self.conn.commit()
        return self.run_id

    def log_model_meta(self, run_id: str, ae_id: str, input_dim: int,
                       model_arch: Dict[str, Any], speed_mode: bool = False,
                       file_name: Optional[str] = None) -> None:
        self.cursor.execute("""
            INSERT INTO runs (
                run_id, ae_id, file_name, timestamp,
                input_dim, model_arch, speed_mode
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            run_id,
            ae_id,
            file_name,
            datetime.now(),
            input_dim,
            json.dumps(model_arch),
            speed_mode
        ))
        self.conn.commit()

    def log_field_stats(self, run_id: str, ae_id: str,
                        field_mse: List[float],
                        raw_stats: Dict[str, Dict],
                        norm_stats: Dict[str, Dict],
                        field_names: List[str]) -> None:
        stat_types = {"raw": raw_stats, "normalized": norm_stats}

        for stat_type, stats in stat_types.items():
            for i, field_name in enumerate(field_names):
                stat_dict = stats.get(field_name)
                if stat_dict is None:
                    continue

                mse_value = float(field_mse[i]) if i < len(field_mse) else None

                self.cursor.execute(
                    """
                    INSERT INTO field_stats (run_id, ae_id, field_index, field_name, stat_type,
                                             min, p5, median, mean, p95, max, mode, mode_count, mse)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id, ae_id, i, field_name, stat_type,
                        stat_dict.get("min"), stat_dict.get("p5"), stat_dict.get("median"),
                        stat_dict.get("mean"), stat_dict.get("p95"), stat_dict.get("max"),
                        stat_dict.get("mode"), stat_dict.get("mode_count"), mse_value
                    )
                )
        self.conn.commit()

    def log_field_errors(self, run_id: str, ae_id: str, epoch_number: int,
                         field_mse: List[float], field_names: List[str]) -> None:
        for field_index, mse in enumerate(field_mse):
            field_name = field_names[field_index]
            self.cursor.execute("""
                INSERT INTO field_errors (run_id, ae_id, field_index, field_name, mse, epoch_number)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                run_id,
                ae_id,
                field_index,
                field_name,
                float(mse),
                epoch_number
            ))
        try:
            self.conn.commit()
            log_with_tag(self.logger, logging.DATABASE, "PostgresLogger",
                         f"[FieldErrors] Committed {len(field_mse)} rows to field_errors for ae_id={ae_id}, epoch={epoch_number}")
            log_with_tag(self.logger, logging.DATABASE, "PostgresLogger",
                         f"[Cursor Status] rowcount={self.cursor.rowcount}, status={self.cursor.statusmessage}")
        except Exception as e:
            log_with_tag(self.logger, logging.ERROR, "PostgresLogger", f"[FieldErrors] Commit Error: {e}")

    def log_sample(self, input_vec: List[float], output_vec: List[float],
                   run_id: str, input_data: Optional[Dict] = None,
                   output_data: Optional[Dict] = None, ae_id: Optional[str] = None) -> None:
        self.cursor.execute("""
            INSERT INTO samples (run_id, input_vec, output_vec, input_data, output_data)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            run_id,
            json.dumps(input_vec),
            json.dumps(output_vec),
            json.dumps(input_data) if input_data else None,
            json.dumps(output_data) if output_data else None
        ))
        self.conn.commit()

    def log_histogram(self, bin_edges: List[float], bin_counts: List[int], ae_id: Optional[str] = None) -> None:
        self.cursor.execute("""
            INSERT INTO reconstruction_histograms (run_id, bin_edges, bin_counts)
            VALUES (%s, %s, %s)
        """, (
            self.run_id,
            json.dumps(bin_edges),
            json.dumps(bin_counts)
        ))
        self.conn.commit()

    def log_model_weights(self, run_id: str, ae_id: str, compressed_weights: bytes) -> None:
        self.cursor.execute("""
            INSERT INTO model_weights (run_id, ae_id, layer_name, weights_compressed)
            VALUES (%s, %s, %s, %s)
        """, (
            run_id,
            ae_id,
            'full_model_blob',
            psycopg2.Binary(compressed_weights)
        ))
        self.conn.commit()

    def update_run_metrics(self,
                           final_loss: Optional[float] = None,
                           best_epoch: Optional[int] = None,
                           mse: Optional[float] = None,
                           reconstructed_mse: Optional[float] = None,
                           reconstructed_cosine: Optional[float] = None,
                           ae_id: Optional[str] = None,
                           run_id: Optional[str] = None) -> None:
        fields = []
        values = []

        if final_loss is not None:
            fields.append("final_loss = %s")
            values.append(float(final_loss))
        if best_epoch is not None:
            fields.append("best_epoch = %s")
            values.append(int(best_epoch))
        if mse is not None:
            fields.append("mse = %s")
            values.append(float(mse))
        if reconstructed_mse is not None:
            fields.append("reconstructed_mse = %s")
            values.append(float(reconstructed_mse))
        if reconstructed_cosine is not None:
            fields.append("reconstructed_cosine = %s")
            values.append(float(reconstructed_cosine))

        if not fields:
            return

        query = f"UPDATE runs SET {', '.join(fields)} WHERE run_id = %s AND ae_id = %s"
        values.extend([run_id, ae_id])
        self.cursor.execute(query, tuple(values))
        self.conn.commit()

    def log_epoch(self, epoch_num: int, layer_stats: Dict[str, Dict[str, Optional[float]]],
                  run_id: Optional[str] = None, ae_id: Optional[str] = None) -> None:
        if not layer_stats:
            if self.logger:
                log_with_tag(self.logger, logging.WARNING, "PostgresLogger",
                             f"[Epoch] No layer_stats provided for ae_id={ae_id}, epoch={epoch_num}")
            return

        for layer, stats in layer_stats.items():
            self.cursor.execute("""
                INSERT INTO epochs (
                    run_id, ae_id, epoch_num, layer_name,
                    grad_mean, grad_std, weight_mean, weight_std, insert_ts
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                run_id,
                ae_id,
                int(epoch_num),
                layer,
                float(stats.get('grad_mean')) if stats.get('grad_mean') is not None else None,
                float(stats.get('grad_std')) if stats.get('grad_std') is not None else None,
                float(stats.get('weight_mean')) if stats.get('weight_mean') is not None else None,
                float(stats.get('weight_std')) if stats.get('weight_std') is not None else None,
                int(time.time_ns())
            ))

        try:
            self.conn.commit()
            log_with_tag(self.logger, logging.DATABASE, "PostgresLogger",
                         f"[LogEpoch] Logged {len(layer_stats)} layers for ae_id={ae_id}, epoch={epoch_num}")
            log_with_tag(self.logger, logging.DATABASE, "PostgresLogger",
                         f"[Cursor Status] rowcount={self.cursor.rowcount}, status={self.cursor.statusmessage}")
        except Exception as e:
            log_with_tag(self.logger, logging.ERROR, "PostgresLogger", f"[LogEpoch] Commit Error: {e}")

    def log_protocol_distribution(
            self,
            stat_type: str,
            stats_dict: Dict[str, Any],
            run_id: Optional[str] = None,
            ae_id: Optional[str] = None
    ) -> None:
        """
        Logs the protocol distribution stats into the database.

        Parameters:
            stat_type: 'raw' or 'normalized'
            stats_dict: dict containing a key 'protocol_distribution' -> {protocol: percentage}
            run_id: optional override for the run ID (defaults to self.run_id)
            ae_id: optional Autoencoder ID
        """
        distribution = stats_dict.get("protocol_distribution")
        if not distribution:
            if self.logger:
                log_with_tag(self.logger, logging.WARNING, "PostgresLogger", "[ProtocolDist] No distribution found")
            return

        actual_run_id = run_id or self.run_id

        for protocol, percent in distribution.items():
            self.cursor.execute(
                """
                INSERT INTO protocol_distribution (run_id, ae_id, stat_type, protocol, percentage)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (actual_run_id, ae_id, stat_type, protocol, float(percent))
            )

        self.conn.commit()
        if self.logger:
            log_with_tag(self.logger, logging.DATABASE, "PostgresLogger",
                         f"[ProtocolDist] Logged {len(distribution)} entries for stat_type={stat_type}, ae_id={ae_id}")

    def close(self) -> None:
        self.cursor.close()
        self.conn.close()