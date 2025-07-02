# networkae/core/evaluation.py

from typing import Optional
import torch
import pandas as pd
from torch.utils.data import TensorDataset

from networkae.model.reporting import PostgresLogger
from networkae.model.model import NetworkAutoencoder


def evaluate_and_log_sample(
    model_wrapper: NetworkAutoencoder,
    db_logger: PostgresLogger,
    run_id: str,
    sample_row: pd.Series,
    field_names: Optional[list[str]] = None
) -> None:
    """
    Evaluates a single sample row and logs reconstruction metrics and vectors.

    Args:
        model_wrapper: Trained NetworkAutoencoder
        db_logger: Database logger for persistence
        run_id: Run ID to associate with logged sample
        sample_row: Pandas Series representing one sample row
        field_names: Optional list of input fields (not used here directly)
    """
    sample_tensor = torch.tensor(sample_row.values, dtype=torch.float32).unsqueeze(0).to(model_wrapper.device)
    mse, cosine_sim = model_wrapper.evaluate(sample_tensor)

    db_logger.update_run_metrics(
        reconstructed_mse=mse,
        reconstructed_cosine=cosine_sim
    )

    model_wrapper.model.eval()
    with torch.no_grad():
        output_tensor = model_wrapper.model(sample_tensor)
        reconstructed = output_tensor.cpu().numpy().flatten().tolist()
        original = sample_tensor.cpu().numpy().flatten().tolist()

    raw_data_dict = sample_row.to_dict()

    db_logger.log_sample(
        run_id=run_id,
        input_vec=original,
        output_vec=reconstructed,
        input_data=raw_data_dict,
        output_data=None
    )


def log_top_sample_errors(
    df: pd.DataFrame,
    dataset: TensorDataset,
    model: NetworkAutoencoder,
    db_logger: PostgresLogger,
    top_n: int = 10
) -> None:
    """
    Logs the top-N samples with the highest reconstruction error.

    Args:
        df: Original DataFrame used for training
        dataset: Tensor dataset corresponding to df
        model: Trained NetworkAutoencoder
        db_logger: Database logger for sample storage
        top_n: Number of top error samples to log
    """
    model.model.eval()
    with torch.no_grad():
        full_tensor = dataset[:][0].to(model.device)
        output_tensor = model.model(full_tensor)

        errors = torch.mean((output_tensor - full_tensor) ** 2, dim=1)
        available = errors.shape[0]
        top_n = min(top_n, available)

        if available < top_n:
            print(f"[log_top_sample_errors] Requested top_n={top_n}, but only {available} samples available. Clamping.")

        top_indices = torch.topk(errors, top_n).indices.cpu().numpy()

        for i in top_indices:
            input_vec = full_tensor[i].cpu().numpy().tolist()
            output_vec = output_tensor[i].cpu().numpy().tolist()
            raw_data = df.iloc[i].to_dict()
            recon_error = float(errors[i].cpu())

            db_logger.log_sample(
                run_id=model.run_id,
                input_vec=input_vec,
                output_vec=output_vec,
                input_data={**raw_data, "reconstruction_error": recon_error},
                output_data=None,
                ae_id=model.ae_id
            )