from typing import Any, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from networkae.model.reporting import PostgresLogger
from networkae.model.evaluation import log_top_sample_errors, evaluate_and_log_sample
from networkae.model.model import NetworkAutoencoder


def log_post_training_summary(
    df: pd.DataFrame,
    dataset: TensorDataset,
    model: NetworkAutoencoder,
    db_logger: PostgresLogger,
    run_id: str,
    sample_row: pd.Series,
    field_names: List[str],
    top_n: int = 10
) -> None:
    """
    Logs post-training evaluations including top error samples, reconstruction stats,
    and histogram of per-row reconstruction errors.

    Args:
        df: Raw DataFrame used during training (for sample reference)
        dataset: Torch dataset for reconstruction evaluation
        model: Trained NetworkAutoencoder instance
        db_logger: PostgresLogger for database persistence
        run_id: Unique run identifier
        sample_row: One row of input data to evaluate and log
        field_names: List of field names for sample vectorization
        top_n: Number of top error rows to log
    """
    log_top_sample_errors(
        df=df,
        dataset=dataset,
        model=model,
        db_logger=db_logger,
        top_n=top_n
    )

    evaluate_and_log_sample(
        model_wrapper=model,
        db_logger=db_logger,
        run_id=run_id,
        sample_row=sample_row,
        field_names=field_names
    )

    model.model.eval()
    with torch.no_grad():
        full_tensor = dataset[:][0].to(model.device)
        output_tensor = model.model(full_tensor)
        errors = torch.mean((output_tensor - full_tensor) ** 2, dim=1).cpu().numpy()

    bin_counts, bin_edges = np.histogram(errors, bins=50)
    db_logger.log_histogram(bin_edges.tolist(), bin_counts.tolist())