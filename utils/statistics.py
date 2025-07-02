import pandas as pd
from typing import Optional, Dict, Any

def compute_column_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    stats = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            mode = series.mode()
            stats[col] = {
                "min": float(series.min()),
                "p5": float(series.quantile(0.05)),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
                "mode": float(mode.iloc[0]) if not mode.empty else None,
                "mode_count": int((series == mode.iloc[0]).sum()) if not mode.empty else 0,
            }
    return stats

def compute_protocol_distribution(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if "protocol" not in df.columns:
        return None
    dist = df["protocol"].value_counts(normalize=True).to_dict()
    return {str(k): round(v, 4) for k, v in dist.items()}

def compute_stats_with_protocol(df: pd.DataFrame) -> Dict[str, Any]:
    stats = compute_column_stats(df)
    protocol_dist = compute_protocol_distribution(df)
    stats["protocol_distribution"] = protocol_dist
    return stats