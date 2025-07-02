import pandas as pd
from typing import List


def get_field_names(df: pd.DataFrame) -> List[str]:
    """
    Returns a list of all column names in the given DataFrame.

    This function makes no exclusions — it is intended for use in
    contexts where all numeric fields (already preprocessed) should
    be passed into a model.

    Args:
        df: Input DataFrame

    Returns:
        List of column names as strings
    """
    return list(df.columns)