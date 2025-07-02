# networkae/entrypoints/run_parallel.py

import os
import yaml
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from networkae.cli.run_single import main as run_single_main


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_file_on_config(file_path, base_config, log_path):
    config = base_config.copy()
    config['input_path'] = file_path
    config['logging_config'] = log_path
    run_single_main(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, help='Directory containing CSV files')
    parser.add_argument('--config', required=True, help='Path to base YAML config file')
    parser.add_argument('--log', default='config/logging.yaml', help='Logger config path')
    parser.add_argument('--max-workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    base_config = load_config(args.config)
    input_files = [os.path.join(args.input_dir, f)
                   for f in os.listdir(args.input_dir)
                   if f.endswith('.csv')]

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_file_on_config, file_path, base_config, args.log)
                   for file_path in input_files]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error during processing: {e}")


if __name__ == '__main__':
    main()
