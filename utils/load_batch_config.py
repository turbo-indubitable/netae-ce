import csv
import json
import psycopg2
from datetime import datetime
import argparse

def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"} if value else False

def load_batch_config(csv_path, db_conn):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            run_name = row["run_name"].strip()
            input_file = row["input_file"].strip()

            config = {
                "training": {
                    "batch_size": int(row.get("batch_size")) if row.get("batch_size") else None,
                    "num_epochs": int(row.get("num_epochs")) if row.get("num_epochs") else None,
                    "floor": float(row.get("floor")) if row.get("floor") else None,
                    "patience": int(row.get("patience")) if row.get("patience") else None,
                    "speed_mode": parse_bool(row.get("speed_mode")),
                    "use_hw_autodetect": parse_bool(row.get("hw_auto")),
                    "seed": int(s) if (s := row.get("seed")) and int(s) != 0 else None,
                }
            }

            with db_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO batch_runs (
                        run_name, input_file, config, status, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, 'pending', %s, %s)
                """, (
                    run_name,
                    input_file,
                    json.dumps(config),
                    datetime.now(),
                    datetime.now()
                ))
                db_conn.commit()

def connect_to_db():
    return psycopg2.connect(
        host="localhost",
        dbname="networkmodel",
        user="nmodel_user",
        password="Hamstring-Blinker-Drastic-Revivable-414@",
        port=5432
    )

def main():
    parser = argparse.ArgumentParser(description="Load batch training configs from a CSV into the database.")
    parser.add_argument("csv_path", help="Path to the batch config CSV file")
    args = parser.parse_args()

    db_conn = connect_to_db()
    load_batch_config(args.csv_path, db_conn)
    print("✅ Batch config loaded successfully.")
    db_conn.close()

if __name__ == "__main__":
    main()