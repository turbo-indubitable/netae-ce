import pandas as pd
import numpy as np
import os


def merge_csv_files(file_paths, output_file):
    # Load all CSV files
    dfs = [pd.read_csv(file) for file in file_paths]

    # Ensure all timestamps are in datetime format
    for df in dfs:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop any rows with invalid timestamps
    dfs = [df.dropna(subset=['timestamp']) for df in dfs]

    # Identify the reference time range (file with the most recent timestamps)
    reference_df = max(dfs, key=lambda df: df['timestamp'].max())

    ref_min_time = reference_df['timestamp'].min()
    ref_max_time = reference_df['timestamp'].max()
    ref_time_range = (ref_max_time - ref_min_time).total_seconds()

    merged_df = reference_df.copy()

    for df in dfs:
        if df is reference_df:
            continue

        # Scale timestamps to fit the reference time range
        src_min_time = df['timestamp'].min()
        src_max_time = df['timestamp'].max()
        src_time_range = (src_max_time - src_min_time).total_seconds()

        # Scale timestamps to match reference time range
        df['timestamp'] = ref_min_time + (df['timestamp'] - src_min_time) * (ref_time_range / src_time_range)

        # Randomize microseconds/nanoseconds to avoid duplicate timestamps
        df['timestamp'] = df['timestamp'] + pd.to_timedelta(np.random.randint(0, 1000000, size=len(df)), unit='us')

        # Append to merged dataframe
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    # Sort by timestamp to mix everything together
    merged_df = merged_df.sort_values(by='timestamp').reset_index(drop=True)

    # Save to CSV
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Merged file saved to: {output_file}")


if __name__ == "__main__":
    # 🔥 Update with actual file paths
    file_paths = [
        'path/to/your/file1.csv',
        'path/to/your/file2.csv',
        'path/to/your/file3.csv'
    ]

    output_file = 'merged_output.csv'

    # Validate file existence
    for file in file_paths:
        if not os.path.exists(file):
            raise FileNotFoundError(f"❌ File not found: {file}")

    merge_csv_files(file_paths, output_file)
