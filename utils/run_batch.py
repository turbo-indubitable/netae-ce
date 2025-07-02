import traceback
import numpy as np
import torch
import random
import psycopg2
import sys
import os

from networkae.model.model import NetworkAutoencoder
from networkae.ingest.ingest import load_data, create_dataset
from networkae.utils.logging_manager import LoggingManager
from networkae.db import initialize_db_logger
from networkae.model.post_training import log_post_training_summary
from networkae.model.reporting import PostgresLogger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def connect_to_db():
    return psycopg2.connect(
        host="localhost",
        dbname="networkmodel",
        user="nmodel_user",
        password="Hamstring-Blinker-Drastic-Revivable-414@",
        port=5432
    )

def run_batch_job(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT id, run_name, input_file, config
        FROM batch_runs
        WHERE status = 'pending'
        ORDER BY id ASC
    """)
    rows = cursor.fetchall()

    for batch_id, run_name, input_path, config in rows:
        try:
            print(f"\U0001F680 Starting run: {run_name}")
            training_config = config["training"]
            speed_mode = training_config.get("speed_mode", False)

            # Set deterministic seed if specified
            seed = training_config.get("seed")
            if seed is not None:
                print(f"[SEED] Setting random seed: {seed}")
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            else:
                print("[SEED] No seed specified — using random state")


            # Load and truncate data if speed_mode
            df, raw_stats, norm_stats = load_data(input_path)
            if speed_mode:
                df = df.head(100)

            dataset, sample_row = create_dataset(df)
            input_dim = df.shape[1]
            field_names = df.columns.tolist()

            # Setup logger
            logger = LoggingManager().get_logger(run_name)
            db_logger = initialize_db_logger({
                "db": {
                    "host": "localhost",
                    "dbname": "networkmodel",
                    "user": "nmodel_user",
                    "password": "Hamstring-Blinker-Drastic-Revivable-414@",
                    "port": 5432
                }
            })

            # Initialize model
            model = NetworkAutoencoder(
                input_dim=input_dim,
                config=training_config,
                db_logger=db_logger,
                file_name=os.path.basename(input_path),
                field_names=field_names,
                speed_mode=speed_mode
            )

            run_id = model.run_id

            # Update run_id + mark as running
            with db_conn.cursor() as update_cur:
                update_cur.execute("""
                    UPDATE batch_runs
                    SET run_id = %s, status = 'running', updated_at = NOW()
                    WHERE id = %s
                """, (run_id, batch_id))
                db_conn.commit()

            # Logging input data metrics
            db_logger.log_field_stats([0.0] * input_dim, raw_stats, norm_stats)
            db_logger.log_protocol_distribution("normalized", norm_stats)
            db_logger.log_protocol_distribution("raw", raw_stats)

            # Start training
            model.train(dataset)

            # log the config parameters with the run
            if model.run_id:
                db_logger.log_training_config(model.run_id, training_config)

            # Post-training output
            log_post_training_summary(
                df=df,
                dataset=dataset,
                model=model,
                db_logger=db_logger,
                sample_row=sample_row,
                field_names=field_names,
                top_n=10
            )

            # Mark as complete
            with db_conn.cursor() as cur2:
                cur2.execute("""
                    UPDATE batch_runs
                    SET status = 'complete', updated_at = NOW()
                    WHERE id = %s
                """, (batch_id,))
                db_conn.commit()

        except Exception as e:
            print(f"\u274C Failed run {run_name}: {e}")
            traceback.print_exc()
            with db_conn.cursor() as fail_cur:
                fail_cur.execute("""
                    UPDATE batch_runs
                    SET status = 'failed', updated_at = NOW()
                    WHERE id = %s
                """, (batch_id,))
                db_conn.commit()

if __name__ == "__main__":
    conn = connect_to_db()
    run_batch_job(conn)
    conn.close()
    print("\u2705 Batch run complete.")