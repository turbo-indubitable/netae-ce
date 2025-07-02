import yaml
import pandas as pd
import hashlib
from typing import List, Dict, Callable, Union


def load_routed_patterns_from_config(filepath: str) -> List[Dict[str, str]]:
    """
    Loads routed traffic pattern definitions from a YAML config file.
    Looks for the key: 'routed_traffic_patterns'

    Args:
        filepath: Path to the YAML configuration file.

    Returns:
        List of pattern dictionaries.
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('routed_traffic_patterns', [])


def validate_filter_expr(expr: str) -> bool:
    """
    Attempts to compile and apply a filter expression using pandas.query().

    Args:
        expr: The filter expression string to validate.

    Returns:
        True if valid, False otherwise.
    """
    try:
        test_df = pd.DataFrame({
            'protocol': ['TCP', 'UDP'],
            'src_port': [443, 53],
            'dst_port': [443, 80],
            'src_ip': ['1.2.3.4', '10.0.0.1'],
            'dst_ip': ['8.8.8.8', '1.1.1.1']
        })
        test_df.query(expr, engine='python')
        return True
    except Exception as e:
        print(f"[Validation Error] '{expr}' → {e}")
        return False


def validate_all_patterns(patterns: List[Dict[str, str]]) -> bool:
    """
    Validates all pattern definitions in the provided list.

    Args:
        patterns: List of pattern dictionaries.

    Returns:
        True if all expressions are valid; False if any are invalid.
    """
    all_valid = True
    for pattern in patterns:
        expr = pattern['filter_expr']
        if not validate_filter_expr(expr):
            print(f"❌ Invalid filter for pattern '{pattern['pattern_id']}'")
            all_valid = False
    return all_valid


def parse_pattern_filters(
        patterns: List[Dict[str, str]]
) -> List[Dict[str, Union[str, Callable[[pd.DataFrame], pd.Index]]]]:
    """
    Parses pattern filters into compiled match functions with attached ae_id.

    Args:
        patterns: List of pattern definitions with 'pattern_id' and 'filter_expr'.

    Returns:
        List of parsed patterns with 'match_fn' and 'ae_id'.
    """
    parsed = []
    for pattern in patterns:
        expr = pattern['filter_expr']
        pattern_id = pattern['pattern_id']
        ae_id = pattern.get("ae_id") or hashlib.sha256(pattern_id.encode()).hexdigest()[:16]

        def make_filter(expression: str) -> Callable[[pd.DataFrame], pd.Index]:
            return lambda df: df.query(expression, engine='python').index

        parsed.append({
            'pattern_id': pattern_id,
            'filter_expr': expr,
            'ae_id': ae_id,
            'match_fn': make_filter(expr)
        })

    return parsed