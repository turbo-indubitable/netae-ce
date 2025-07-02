# networkae/core/early_stopping.py

import numpy as np
import logging
from typing import Optional, Tuple, Union


class NullLogger:
    def log(self, level, msg): pass
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def log_with_tag(logger: Optional[logging.Logger], level: int, tag: str, msg: str) -> None:
    if not logger or not hasattr(logger, "log"):
        print(f"[{tag}] {msg}")
    else:
        logger.log(level, f"[{tag}] {msg}")


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-5,
        floor: float = 0.01,
        rolling_window: int = 4,
        min_improvement: float = 0.2,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.logger = logger or NullLogger()

        self.patience = patience
        self.min_delta = min_delta if min_delta is not None else 1e-5
        self.floor = floor
        self.min_improvement = min_improvement
        self.rolling_window = rolling_window

        self.counter = 0
        self.best_loss = float('inf')
        self.loss_history: list[float] = []
        self.improvement_threshold: Optional[float] = None
        self._rolling_avg: Optional[float] = None

        log_with_tag(self.logger, logging.INFO, "EarlyStopping", f"Initialized with min_delta={min_delta}")
        log_with_tag(self.logger, logging.DEBUG, "EarlyStopping",
                     f"Patience: {self.patience} (from config.early_stopping.patience)")

    def patience_left(self) -> int:
        return max(0, self.patience - self.counter)

    def __call__(self, epoch_loss: float) -> bool:
        if epoch_loss < self.floor:
            log_with_tag(self.logger, logging.INFO, "EarlyStopping",
                         f"Stopping early — Epoch loss hit floor: {epoch_loss:.6f} < {self.floor:.6f}")
            return True

        self.loss_history.append(epoch_loss)
        if len(self.loss_history) > self.rolling_window:
            self.loss_history.pop(0)

        if len(self.loss_history) == self.rolling_window:
            self._rolling_avg = float(np.mean(self.loss_history))
            self.improvement_threshold = self._rolling_avg * (1 - self.min_improvement)

            if epoch_loss > self.improvement_threshold:
                self.counter += 1
            else:
                self.counter = max(0, self.counter - 1)

        if epoch_loss < (self.best_loss - self.min_delta):
            self.best_loss = epoch_loss
            self.counter = 0

        if self.counter >= self.patience:
            log_with_tag(self.logger, logging.INFO, "EarlyStopping",
                         f"Early stopping triggered after {self.patience} epochs without improvement.")
            return True

        return False

    def get_metrics(self) -> Tuple[Union[str, float], Union[str, float]]:
        if self._rolling_avg is None or self.improvement_threshold is None:
            return "–", "–"
        return f"{self._rolling_avg:.2f}", f"{self.improvement_threshold:.2f}"