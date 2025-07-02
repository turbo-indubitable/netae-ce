# networkae/core/ingest.py

import ipaddress
import torch
from torch.utils.data import TensorDataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from networkae.utils.transform import split_ip
from networkae.utils.statistics import compute_stats_with_protocol


def split_ip(ip):
    try:
        if pd.isna(ip) or ip == '':
            raise ValueError("Empty IP value")

        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.version == 4:
            return list(map(int, ip.split('.'))) + [0] * 4
        elif ip_obj.version == 6:
            hextets = [(int(h, 16) if h else 0) for h in ip.split(':')]
            while len(hextets) < 8:
                hextets.insert(hextets.index(0), 0)
            return hextets
    except Exception:
        return [0] * 8


TCP_FLAG_MAP = {
    'F': 0x01,
    'S': 0x02,
    'R': 0x04,
    'P': 0x08,
    'A': 0x10,
    'U': 0x20,
    'E': 0x40,
    'C': 0x80
}

def convert_tcp_flags(flags):
    try:
        if pd.isna(flags) or flags == '':
            return 0
        return sum(TCP_FLAG_MAP.get(char, 0) for char in flags)
    except Exception:
        return 0


def compute_statistics(raw_df=None, normalized_df=None):
    stats = {}
    numeric_columns = ['src_port', 'dst_port', 'packet_size', 'ttl', 'icmp_type', 'icmp_code']

    if raw_df is not None:
        for col in numeric_columns:
            if col in raw_df.columns:
                mode = raw_df[col].mode()
                stats[col] = {
                    'min': round(float(raw_df[col].min()), 4),
                    'p5': round(float(raw_df[col].quantile(0.05)), 4),
                    'median': round(float(raw_df[col].median()), 4),
                    'mean': round(float(raw_df[col].mean()), 4),
                    'p95': round(float(raw_df[col].quantile(0.95)), 4),
                    'max': round(float(raw_df[col].max()), 4),
                    'mode': round(float(mode.iloc[0]), 4) if not mode.empty else None,
                    'mode_count': int((raw_df[col] == mode.iloc[0]).sum()) if not mode.empty else 0
                }
        if 'protocol' in raw_df.columns:
            dist = raw_df['protocol'].value_counts(normalize=True).to_dict()
            stats['protocol_distribution'] = {str(k): round(v, 4) for k, v in dist.items()}
        else:
            stats['protocol_distribution'] = None

    if normalized_df is not None:
        for col in numeric_columns:
            if col in normalized_df.columns:
                mode = normalized_df[col].mode()
                stats[col] = {
                    'min': round(float(normalized_df[col].min()), 4),
                    'p5': round(float(normalized_df[col].quantile(0.05)), 4),
                    'median': round(float(normalized_df[col].median()), 4),
                    'mean': round(float(normalized_df[col].mean()), 4),
                    'p95': round(float(normalized_df[col].quantile(0.95)), 4),
                    'max': round(float(normalized_df[col].max()), 4),
                    'mode': round(float(mode.iloc[0]), 4) if not mode.empty else None,
                    'mode_count': int((normalized_df[col] == mode.iloc[0]).sum()) if not mode.empty else 0
                }
        if 'protocol' in normalized_df.columns:
            dist = normalized_df['protocol'].value_counts(normalize=True).to_dict()
            stats['protocol_distribution'] = {str(k): round(v, 4) for k, v in dist.items()}
        else:
            stats['protocol_distribution'] = None

    return stats


def load_data(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip')

    # Compute raw stats before mutation
    raw_stats = compute_stats_with_protocol(df)

    df = df.drop(columns=['file_name'], errors='ignore')
    df = df.sort_values(by='timestamp')

    # Split IPs
    for col in ['src_ip', 'dst_ip']:
        ip_split = df[col].apply(split_ip)
        df[[f'{col}_{i}' for i in range(8)]] = pd.DataFrame(ip_split.tolist(), index=df.index)

    df = df.drop(columns=['src_ip', 'dst_ip'])

    for i in range(8):
        df[f'src_ip_{i}'] = df[f'src_ip_{i}'].astype('float32')
        df[f'dst_ip_{i}'] = df[f'dst_ip_{i}'].astype('float32')

    df['timestamp'] = df['timestamp'] / 1e9
    df['protocol'] = pd.to_numeric(df['protocol'], errors='coerce').fillna(-1)

    # Gate fields based on protocol
    df['src_port'] = df.apply(lambda row: row['src_port'] if row['protocol'] in [6, 17] else 0, axis=1)
    df['dst_port'] = df.apply(lambda row: row['dst_port'] if row['protocol'] in [6, 17] else 0, axis=1)
    df['icmp_type'] = df.apply(lambda row: row['icmp_type'] if row['protocol'] == 1 else 0, axis=1)
    df['icmp_code'] = df.apply(lambda row: row['icmp_code'] if row['protocol'] == 1 else 0, axis=1)

    # Log-scale numeric fields
    df['src_port'] = np.log1p(df['src_port']) / 5
    df['dst_port'] = np.log1p(df['dst_port']) / 5
    df['packet_size'] = np.log1p(df['packet_size']) / 5

    df = df.fillna(0)

    # Normalize selected columns
    scaler = MinMaxScaler()
    numeric_cols = ['src_ip_0', 'src_ip_1', 'src_ip_2', 'src_ip_3',
                    'dst_ip_0', 'dst_ip_1', 'dst_ip_2', 'dst_ip_3',
                    'src_port', 'dst_port', 'packet_size', 'ttl']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # One-hot encode protocol
    protocol_encoder = OneHotEncoder(sparse_output=False)
    protocol_encoded = protocol_encoder.fit_transform(df[['protocol']])
    protocol_df = pd.DataFrame(protocol_encoded, columns=[f'proto_{i}' for i in range(protocol_encoded.shape[1])])
    df = pd.concat([df.drop(columns=['protocol']), protocol_df], axis=1).astype('float32')

    # Compute post-processing stats
    norm_stats = compute_stats_with_protocol(df)

    return df, raw_stats, norm_stats


def create_dataset(data):
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    tensor_data = torch.tensor(data.values, dtype=torch.float32)
    dataset = TensorDataset(tensor_data, tensor_data)
    return dataset, data.iloc[0].copy()  # For reconstruction comparison
