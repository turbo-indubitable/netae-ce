import io
import gzip
import sys
import logging
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from networkae.model.early_stopping import EarlyStopping
from networkae.utils.statistics import compute_stats_with_protocol


class NullLogger:
    def log(self, level, msg): pass
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
    def database(self, msg): pass  # Custom level
    def train(self, msg): pass     # Custom level
    def summary(self, msg): pass   # Custom level

def log_with_tag(logger, level, tag, message):
    if not hasattr(logger, "log") or not callable(logger.log):
        print(f"[{tag}] {message}")
        return
    logger.log(level, f"[{tag}] {message}")


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder_input = nn.Linear(input_dim, 64)
        self.encoder_hidden = nn.Linear(64, 32)
        self.encoder_latent = nn.Linear(32, 16)
        self.decoder_hidden = nn.Linear(16, 32)
        self.decoder_expand = nn.Linear(32, 64)
        self.decoder_output = nn.Linear(64, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.encoder_input(x))
        x = torch.relu(self.encoder_hidden(x))
        latent = self.encoder_latent(x)
        x = torch.relu(self.decoder_hidden(latent))
        x = torch.relu(self.decoder_expand(x))
        return self.decoder_output(x)


class NetworkAutoencoder:
    def __init__(
        self,
        input_dim: int,
        config: dict,
        early_stopping_config: dict,
        db_logger=None,
        file_name=None,
        field_names=None,
        speed_mode=False,
        logger=None,
        run_id=None,
        ae_id=None
    ):
        self.speed_mode = speed_mode
        self.input_dim = input_dim
        self.config = config
        self.early_stopping_config = early_stopping_config
        self.db_logger = db_logger
        self.file_name = file_name
        self.field_names = field_names or []
        self.run_id = run_id
        self.ae_id = ae_id
        self.field_mse: List[float] = []  # ← Add this line
        self.raw_stats: Dict[str, Dict[str, float]] = {}
        self.normalized_stats: Dict[str, Dict[str, float]] = {}
        self.gradient_logs: List[Dict[str, float]] = []
        self.logger = logger or NullLogger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log_with_tag(self.logger, logging.DEBUG, "Model", f"Using device: {self.device}")
        log_with_tag(self.logger, logging.DEBUG, "Model",
                     f"Learning rate: {self.config.get('learning_rate', 0.001)} (from config.training.learning_rate)")
        log_with_tag(self.logger, logging.DEBUG, "Model",
                     f"Decay factor: {self.config.get('lr_decay_factor')} (from config.training.lr_decay_factor)")
        log_with_tag(self.logger, logging.DEBUG, "Model",
                     f"Patience: {self.config.get('lr_decay_patience')} (from config.training.lr_decay_patience)")
        log_with_tag(self.logger, logging.DEBUG, "Model",
                     f"Early stopping: {self.config.get('early_stopping')} (from config.early_stopping)")

        self.model = Autoencoder(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get("learning_rate", 0.001))
        self.criterion = nn.MSELoss()
        self.early_stopper = EarlyStopping(**early_stopping_config, logger=self.logger)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get("lr_decay_factor", 0.5),
            patience=config.get("lr_decay_patience", 3),
        )

        self.gradient_logs: List[Dict[str, float]] = []
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_weights = None

        log_with_tag(self.logger, logging.INFO, "Model", f"Model settings: {file_name}, Speed mode: {speed_mode}")

        if db_logger and run_id and file_name:
            model_arch = {name: list(param.shape) for name, param in self.model.named_parameters()}
            db_logger.log_model_meta(
                run_id=self.run_id,
                ae_id=self.ae_id,
                file_name=file_name,
                input_dim=input_dim,
                model_arch=model_arch,
                speed_mode=speed_mode
            )

    def train(self, dataset: TensorDataset, batch_size: Optional[int] = None) -> None:
        """
        Train the autoencoder model on a dataset with early stopping and logging support.

        Args:
            dataset (TensorDataset): Input dataset containing (features, features).
            batch_size (Optional[int]): Optional override for the training batch size.
        """
        effective_batch_size = batch_size or self.config.get("batch_size")
        num_epochs = self.config.get("num_epochs")

        log_with_tag(self.logger, logging.DEBUG, "Model",
                     f"Batch size used: {effective_batch_size} (from config.training.batch_size)")
        log_with_tag(self.logger, logging.DEBUG, "Model",
                     f"Num epochs used: {num_epochs} (from config.training.num_epochs)")

        if num_epochs is None:
            raise ValueError("config.training.num_epochs is missing — no default allowed.")
        if effective_batch_size is None:
            raise ValueError("config.training.batch_size is missing — no default allowed.")

        early_stopping = EarlyStopping(logger=self.logger, **self.early_stopping_config)
        floor = self.early_stopping_config.get("floor", 0.01)
        log_with_tag(self.logger, logging.INFO, "Model",
                     f"Training: batch_size={effective_batch_size}, epochs={num_epochs}, floor={floor}")

        data_loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, pin_memory=True)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            num_samples = 0

            with tqdm(data_loader, colour="white", file=sys.stdout, unit="batch",
                      desc=f"Epoch {epoch + 1}") as progress_bar:
                for batch, _ in progress_bar:
                    batch = batch.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(batch)
                    loss = self.criterion(output, batch)
                    loss.backward()
                    self.optimizer.step()

                    bsize = batch.size(0)
                    epoch_loss += loss.item() * bsize
                    num_samples += bsize

                    avg_loss = epoch_loss / num_samples
                    rolling_avg, threshold = early_stopping.get_metrics()
                    progress_bar.set_postfix({
                        "epoch_loss": f"{avg_loss:.5f}",
                        "rolling_avg": rolling_avg,
                        "threshold": threshold,
                        "patience_left": early_stopping.patience_left()
                    })

            epoch_loss /= num_samples

            if self.db_logger and self.run_id and self.field_names:
                with torch.no_grad():
                    self.model.eval()
                    full_tensor = dataset[:][0].clone().detach().to(self.device)
                    outputs = self.model(full_tensor)
                    field_errors = (outputs - full_tensor) ** 2
                    epoch_field_mse = torch.mean(field_errors, dim=0).cpu().numpy()
                    self.db_logger.log_field_errors(
                        run_id=self.run_id,
                        ae_id=self.ae_id,
                        epoch_number=epoch + 1,
                        field_mse=epoch_field_mse,
                        field_names=self.field_names
                    )

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_epoch = epoch
                self.best_weights = self.model.state_dict()

            stats = self.log_weights_and_gradients()
            self.gradient_logs.append(stats)

            log_with_tag(self.logger, logging.TRAIN, "Model",
                         f"***Epoch logging → run_id={self.run_id}, ae_id={self.ae_id}, epoch={epoch + 1}")

            if self.db_logger and self.run_id:
                self.db_logger.log_epoch(
                    epoch_num=epoch + 1,
                    layer_stats=stats,
                    run_id=self.run_id,
                    ae_id=self.ae_id
                )

            if early_stopping(epoch_loss):
                log_with_tag(self.logger, logging.INFO, "Model", f"[EARLY STOP] epoch_loss = {epoch_loss:.6f}")
                break

        # Post-training model commit
        if self.db_logger and self.run_id:
            self.db_logger.update_run_metrics(
                final_loss=self.best_loss,
                best_epoch=self.best_epoch + 1,
                mse=self.best_loss,
                run_id=self.run_id,
                ae_id=self.ae_id
            )
            self.model.load_state_dict(self.best_weights)
            buffer = io.BytesIO()
            torch.save(self.best_weights, buffer)
            compressed = gzip.compress(buffer.getvalue())
            self.db_logger.log_model_weights(
                run_id=self.run_id,
                ae_id=self.ae_id,
                compressed_weights=compressed
            )

        # Compute and log final stats
        full_tensor = dataset[:][0].cpu()
        raw_df = pd.DataFrame(full_tensor.numpy(), columns=self.field_names)
        scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(raw_df), columns=self.field_names)
        self.raw_stats = compute_stats_with_protocol(raw_df)
        self.normalized_stats = compute_stats_with_protocol(scaled_df)
        with torch.no_grad():
            self.field_mse = torch.mean(
                (self.model(torch.tensor(scaled_df.values, dtype=torch.float32).to(self.device)) - full_tensor.to(
                    self.device)) ** 2,
                dim=0
            ).cpu().numpy().tolist()

        log_with_tag(self.logger, logging.DEBUG, "Model",
                     f"Effective batch size: {effective_batch_size} (from config.training.batch_size)")
        log_with_tag(self.logger, logging.DEBUG, "Model", f"Num epochs: {num_epochs} (from config.training.num_epochs)")

        if self.db_logger and self.run_id:
            self.db_logger.log_field_stats(
                run_id=self.run_id,
                ae_id=self.ae_id,
                field_mse=self.field_mse,
                raw_stats=self.raw_stats,
                norm_stats=self.normalized_stats,
                field_names=self.field_names
            )

    def train_on_chunk(self, df_chunk: pd.DataFrame, bin_start: Optional[str] = None, db_logger=None) -> None:
        if df_chunk.empty:
            return

        input_tensor = torch.tensor(df_chunk[self.field_names].values, dtype=torch.float32)
        dataset = TensorDataset(input_tensor, input_tensor)

        bin_rows = len(df_chunk)
        min_batches = self.config.get("min_batches_per_bin", 4)
        max_batches = self.config.get("max_batches_per_bin", 25)

        batch_target = max(1, bin_rows // max_batches)
        batch_limit = max(1, bin_rows // min_batches)
        adaptive_batch_size = min(batch_limit, batch_target)

        log_with_tag(self.logger, logging.TRAIN, "Model",
                     f"[BatchSize] AE {self.ae_id} | Bin {bin_start} | Rows: {bin_rows} | "
                     f"Target: {batch_target} | Limit: {batch_limit} | Using: {adaptive_batch_size}")
        log_with_tag(self.logger, logging.TRAIN, "Model",
                     f"[Shape] AE {self.ae_id} training on tensor with shape: {input_tensor.shape}")
        log_with_tag(self.logger, logging.TRAIN, "Model",
                     f"[DEBUG] Field names used for training: {self.field_names}")
        log_with_tag(self.logger, logging.DEBUG, "Model",
                     f"min_batches_per_bin: {min_batches} (from config.training.min_batches_per_bin)")
        log_with_tag(self.logger, logging.DEBUG, "Model",
                     f"max_batches_per_bin: {max_batches} (from config.training.max_batches_per_bin)")

        if self.speed_mode:
            self.model.train()
            total_loss = 0
            total_samples = 0
            final_batch = None

            loader = DataLoader(dataset, batch_size=adaptive_batch_size, shuffle=True)
            for batch, _ in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch.size(0)
                total_samples += batch.size(0)
                final_batch = (output.detach(), batch.detach())

            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            cosine_sim = torch.nn.functional.cosine_similarity(*final_batch).mean().item() if final_batch else None

            if final_batch:
                df_raw = pd.DataFrame(final_batch[1].cpu().numpy(), columns=self.field_names)
                df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df_raw), columns=self.field_names)
                self.raw_stats = compute_stats_with_protocol(df_raw)
                self.normalized_stats = compute_stats_with_protocol(df_scaled)

        else:
            self.train(dataset, batch_size=adaptive_batch_size)
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(input_tensor.to(self.device))
                mse_tensor = torch.nn.functional.mse_loss(predictions, input_tensor.to(self.device), reduction='none')
                avg_loss = mse_tensor.mean().item()
                cosine_sim = torch.nn.functional.cosine_similarity(predictions, input_tensor.to(self.device)).mean().item()

            df_raw = pd.DataFrame(input_tensor.cpu().numpy(), columns=self.field_names)
            df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df_raw), columns=self.field_names)
            self.raw_stats = compute_stats_with_protocol(df_raw)
            self.normalized_stats = compute_stats_with_protocol(df_scaled)
            total_samples = len(input_tensor)

        if self.db_logger and bin_start:
            self.db_logger.log_bin_stats(
                ae_id=self.ae_id,
                bin_start=bin_start,
                mse=avg_loss,
                cosine=cosine_sim,
                count=total_samples
            )
            self.db_logger.update_run_metrics(
                reconstructed_mse=avg_loss,
                reconstructed_cosine=cosine_sim,
                run_id=self.run_id,
                ae_id=self.ae_id
            )

        log_with_tag(self.logger, logging.INFO, "Model",
                     f"[Chunk Train] AE {self.ae_id} | Bin {bin_start} | Samples {total_samples} | "
                     f"MSE: {avg_loss:.6f} | CosSim: {cosine_sim:.4f}")

    def evaluate(self, sample_tensor: torch.Tensor) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_tensor.to(self.device))

        mse = torch.mean((output - sample_tensor) ** 2).item()
        cos_sim = torch.nn.functional.cosine_similarity(output, sample_tensor).mean().item()
        return mse, cos_sim

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def log_weights_and_gradients(self) -> Dict[str, Dict[str, Optional[float]]]:
        stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'weight_mean': round(param.data.mean().item(), 6),
                    'weight_std': round(param.data.std().item(), 6),
                    'grad_mean': round(param.grad.mean().item(), 6) if param.grad is not None else None,
                    'grad_std': round(param.grad.std().item(), 6) if param.grad is not None else None
                }
        return stats


def load_model(
    ae_id: str,
    training_config: Optional[Dict[str, Any]] = None,
    early_stopping_config: Optional[Dict[str, Any]] = None,
    db_logger=None,
    field_names: Optional[List[str]] = None,
    file_name: Optional[str] = None,
    speed_mode: bool = False,
    logger=None,
    run_id: Optional[str] = None
) -> NetworkAutoencoder:

    training_config = training_config or {}
    early_stopping_config = early_stopping_config or {}

    field_names = field_names or []
    input_dim = len(field_names)

    print(f"[DEBUG] Model built with input_dim={input_dim} for ae_id={ae_id}")

    return NetworkAutoencoder(
        input_dim=input_dim,
        config=training_config,
        early_stopping_config=early_stopping_config,
        db_logger=db_logger,
        file_name=file_name or ae_id,
        field_names=field_names,
        speed_mode=speed_mode,
        logger=logger,
        run_id=run_id,
        ae_id=ae_id
    )