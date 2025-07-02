import logging
import ipaddress
from typing import List, Optional, Union

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from networkae.utils.logging_manager import log_with_tag


TCP_FLAG_MAP = {
    'F': 0x01,  # FIN
    'S': 0x02,  # SYN
    'R': 0x04,  # RST
    'P': 0x08,  # PUSH
    'A': 0x10,  # ACK
    'U': 0x20,  # URGENT
    'E': 0x40,  # ECE
    'C': 0x80   # CWR
}


def preprocess_for_training_bin(df: pd.DataFrame, expected_proto_cols: Optional[List[str]] = None) -> pd.DataFrame:
    logger = logging.getLogger("networkae")
    log_with_tag(logger, logging.INFO, "Preprocess-Bin", "Beginning bin preprocessing")

    df = df.copy()

    if 'tcp_flags' in df.columns:
        tqdm.pandas(desc="TCP Flag Conversion")
        df['tcp_flags'] = df['tcp_flags'].astype(object).progress_apply(lambda f: convert_tcp_flags(f, logger=logger))

    for ip_field in ['src_ip', 'dst_ip']:
        if ip_field in df.columns:
            tqdm.pandas(desc=f"Split {ip_field}")
            ip_parts = df[ip_field].progress_apply(split_ip)
            df[[f"{ip_field}_{i+1}" for i in range(8)]] = pd.DataFrame(ip_parts.tolist(), index=df.index)

    for col in tqdm(['src_port', 'dst_port', 'packet_size'], desc="Log-Scaling"):
        if col in df.columns:
            df[f'{col}_scaled'] = np.log1p(df[col]) / 5

    ip_cols = [col for col in df.columns if col.startswith('src_ip_') or col.startswith('dst_ip_')]
    scaled_cols = [col for col in df.columns if col.endswith('_scaled')]
    numeric_cols = ip_cols + scaled_cols

    if numeric_cols:
        log_with_tag(logger, logging.DEBUG, "Preprocess-Bin", f"MinMax scaling {len(numeric_cols)} numeric columns")
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    if 'protocol' in df.columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        proto_array = encoder.fit_transform(df[['protocol']])
        proto_df = pd.DataFrame(proto_array, index=df.index,
                                columns=[f'proto_{i}' for i in range(proto_array.shape[1])])
        df = pd.concat([df, proto_df], axis=1)

        if expected_proto_cols:
            for col in expected_proto_cols:
                if col not in df.columns:
                    df[col] = 0.0
        df.drop(columns=['protocol'], inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    return df


def preprocess_for_training_sample(df: pd.DataFrame, expected_proto_cols: Optional[List[str]] = None) -> pd.DataFrame:
    logger = logging.getLogger("networkae")
    log_with_tag(logger, logging.INFO, "Preprocess-Sample", "Beginning sample preprocessing")

    df = df.copy()

    for col in tqdm(['src_port', 'dst_port', 'packet_size'], desc="Log-Scaling"):
        if col in df.columns:
            df[f'{col}_scaled'] = np.log1p(df[col]) / 5

    ip_cols = [col for col in df.columns if col.startswith('src_ip_') or col.startswith('dst_ip_')]
    scaled_cols = [col for col in df.columns if col.endswith('_scaled')]
    numeric_cols = ip_cols + scaled_cols

    if numeric_cols:
        log_with_tag(logger, logging.DEBUG, "Preprocess-Sample", f"MinMax scaling {len(numeric_cols)} numeric fields")
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    if 'protocol' in df.columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        proto_array = encoder.fit_transform(df[['protocol']])
        proto_df = pd.DataFrame(proto_array, index=df.index,
                                columns=[f'proto_{i}' for i in range(proto_array.shape[1])])
        df = pd.concat([df, proto_df], axis=1)

        if expected_proto_cols:
            for col in expected_proto_cols:
                if col not in df.columns:
                    df[col] = 0.0

        df.drop(columns=['protocol'], inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    return df


def convert_tcp_flags(flags: Union[str, float, int], logger: Optional[logging.Logger] = None) -> int:
    try:
        if pd.isna(flags) or flags == '':
            return 0

        if isinstance(flags, (int, float)) and not isinstance(flags, bool):
            return int(flags)

        return sum(TCP_FLAG_MAP.get(char, 0) for char in str(flags))
    except Exception as e:
        log_with_tag(logger, logging.WARN, "Preprocess", f"Failed to convert TCP flags: {flags} → {e}")
        return 0


def split_ip(ip: str) -> List[int]:
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


def preprocess_for_routing(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger("networkae")
    df = df.copy()

    log_with_tag(logger, logging.DEBUG, "Preprocess-Route", "Dropping file_name and normalizing timestamp")

    df = df.drop(columns=[col for col in ['file_name'] if col in df.columns])

    if 'timestamp' in df.columns:
        if pd.api.types.is_numeric_dtype(df['timestamp']):
            df['timestamp'] = df['timestamp'] / 1e9
        elif pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = df['timestamp'].astype('int64') / 1e9

    if 'tcp_flags' in df.columns:
        log_with_tag(logger, logging.DEBUG, "Preprocess-Route", "Converting TCP flags")
        tqdm.pandas(desc="TCP Flag Conversion")
        df['tcp_flags'] = df['tcp_flags'].astype(object).progress_apply(convert_tcp_flags)

    for col in ['src_ip', 'dst_ip']:
        if col in df.columns:
            log_with_tag(logger, logging.DEBUG, "Preprocess-Route", f"Expanding IP column: {col}")
            tqdm.pandas(desc=f"Split {col}")
            ip_expanded = df[col].progress_apply(split_ip)
            df[[f"{col}_{i + 1}" for i in range(8)]] = pd.DataFrame(ip_expanded.tolist(), index=df.index)

    if 'protocol' in df.columns:
        log_with_tag(logger, logging.DEBUG, "Preprocess-Route", "Normalizing protocol-dependent fields")
        df['protocol'] = pd.to_numeric(df['protocol'], errors='coerce').fillna(-1)

        if 'src_port' in df.columns:
            df['src_port'] = tqdm(df.apply(lambda r: r['src_port'] if r['protocol'] in [6, 17] else 0, axis=1),
                                  desc="Gate src_port")
        if 'dst_port' in df.columns:
            df['dst_port'] = tqdm(df.apply(lambda r: r['dst_port'] if r['protocol'] in [6, 17] else 0, axis=1),
                                  desc="Gate dst_port")
        if 'icmp_type' in df.columns:
            df['icmp_type'] = tqdm(df.apply(lambda r: r['icmp_type'] if r['protocol'] == 1 else 0, axis=1),
                                   desc="Gate icmp_type")
        if 'icmp_code' in df.columns:
            df['icmp_code'] = tqdm(df.apply(lambda r: r['icmp_code'] if r['protocol'] == 1 else 0, axis=1),
                                   desc="Gate icmp_code")

    log_with_tag(logger, logging.DEBUG, "Preprocess-Route", "Final coercion and NaN filling")
    df = df.fillna(0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    df['__routing_preprocessed__'] = 1.0

    return df


def preprocess_for_training(df: pd.DataFrame, expected_proto_cols: Optional[List[str]] = None) -> pd.DataFrame:
    logger = logging.getLogger("networkae")
    log_with_tag(logger, logging.INFO, "Preprocess-Train", "Assuming routing stage already complete — skipping.")
    log_with_tag(logger, logging.DEBUG, "Preprocess-Train", "Log-scaling selected fields")

    for col in tqdm(['src_port', 'dst_port', 'packet_size'], desc="Log-Scaling"):
        if col in df.columns:
            df[f'{col}_scaled'] = np.log1p(df[col]) / 5

    ip_cols = [col for col in df.columns if col.startswith('src_ip_') or col.startswith('dst_ip_')]
    scaled_cols = [col for col in df.columns if col.endswith('_scaled')]
    numeric_cols = ip_cols + scaled_cols

    if numeric_cols:
        log_with_tag(logger, logging.DEBUG, "Preprocess-Train", f"MinMax scaling {len(numeric_cols)} columns")
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    if 'protocol' in df.columns:
        log_with_tag(logger, logging.DEBUG, "Preprocess-Train", "One-hot encoding 'protocol' column")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        protocol_encoded = encoder.fit_transform(df[['protocol']])
        encoded_df = pd.DataFrame(protocol_encoded, index=df.index,
                                  columns=[f'proto_{i}' for i in range(protocol_encoded.shape[1])])
        df = pd.concat([df, encoded_df], axis=1)

        if expected_proto_cols:
            for col in expected_proto_cols:
                if col not in df.columns:
                    df[col] = 0.0
        else:
            expected_proto_cols = [f'proto_{i}' for i in range(protocol_encoded.shape[1])]
            for col in expected_proto_cols:
                if col not in df.columns:
                    df[col] = 0.0

        df = df.drop(columns=['protocol'])

    log_with_tag(logger, logging.DEBUG, "Preprocess-Train", "Final coercion to float32")

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df.astype('float32')

    return df