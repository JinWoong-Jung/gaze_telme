"""Logging wrapper — supports W&B and plain console / file logging.

Usage:
    from utils.logger import Logger
    logger = Logger(project="gaze_telme", run_name="phase5_gated", use_wandb=True)
    logger.log({"loss": 0.42, "dev_f1": 0.61}, step=10)
    logger.finish()
"""

import json
import os
from pathlib import Path
from typing import Optional


class Logger:
    """Thin wrapper around wandb (optional) + local JSON file log.

    Args:
        project:   W&B project name
        run_name:  W&B run name
        log_dir:   directory to write run_log.jsonl
        use_wandb: whether to also log to Weights & Biases
        config:    dict of hyperparameters to record at run start
    """

    def __init__(
        self,
        project: str = "gaze_telme",
        run_name: str = "run",
        log_dir: str | Path = "logs",
        use_wandb: bool = False,
        config: Optional[dict] = None,
    ):
        self._use_wandb = use_wandb
        self._run = None
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = log_dir / f"{run_name}_metrics.jsonl"

        if use_wandb:
            try:
                import wandb
                self._run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                    reinit=True,
                )
            except Exception as exc:
                print(f"[Logger] W&B init failed ({exc}). Falling back to file-only.")
                self._use_wandb = False

        if config:
            self._write_local({"event": "config", **config}, step=0)

    def log(self, metrics: dict, step: int = 0) -> None:
        """Log a metrics dict at a given step."""
        self._write_local(metrics, step=step)
        if self._use_wandb and self._run is not None:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def finish(self) -> None:
        if self._use_wandb and self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass

    def _write_local(self, data: dict, step: int) -> None:
        entry = {"step": step, **data}
        with self._log_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")
