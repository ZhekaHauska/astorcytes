import os
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentLogger(ABC):
    @abstractmethod
    def log_params(self, params: dict):
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = None):
        ...

    @abstractmethod
    def log_artifact(self, filepath: str):
        ...

    @abstractmethod
    def finish(self):
        ...


class FileLogger(ExperimentLogger):
    def __init__(self, output_dir: str, experiment_name: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"
        self.exp_dir = self.output_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.params_path = self.exp_dir / "params.json"
        self.metrics_path = self.exp_dir / "metrics.jsonl"
        self._metrics_file = open(self.metrics_path, 'a')
        self._params = {}
        self._metrics = []
        logger.info(f"FileLogger: logging to {self.exp_dir}")

    def log_params(self, params: dict):
        self._params.update(params)
        with open(self.params_path, 'w') as f:
            json.dump(self._params, f, indent=2, default=str)

    def log_metrics(self, metrics: dict, step: int = None):
        entry = {}
        if step is not None:
            entry["step"] = step
        entry.update(metrics)
        entry["timestamp"] = datetime.now().isoformat()
        self._metrics_file.write(json.dumps(entry, default=str) + "\n")
        self._metrics_file.flush()

    def log_artifact(self, filepath: str):
        import shutil
        src = Path(filepath)
        dst = self.exp_dir / src.name
        if src.exists():
            shutil.copy2(src, dst)

    def finish(self):
        self._metrics_file.close()


class CometLogger(ExperimentLogger):
    def __init__(self, output_dir: str = None, experiment_name: str = None,
                 workspace: str = None, project_name: str = None, **kwargs):
        try:
            import comet_ml
        except ImportError:
            raise ImportError("comet-ml is required for CometLogger. Install with: pip install comet-ml")
        self.experiment = comet_ml.Experiment(
            workspace=workspace,
            project_name=project_name,
            **kwargs,
        )
        self._output_dir = output_dir

    def log_params(self, params: dict):
        self.experiment.log_parameters(params)

    def log_metrics(self, metrics: dict, step: int = None):
        self.experiment.log_metrics(metrics, step=step)

    def log_artifact(self, filepath: str):
        self.experiment.log_asset(filepath)

    def finish(self):
        self.experiment.end()


class AimLogger(ExperimentLogger):
    def __init__(self, output_dir: str = None, experiment_name: str = None,
                 repo: str = None, **kwargs):
        try:
            from aim import Run
        except ImportError:
            raise ImportError("aim is required for AimLogger. Install with: pip install aim")
        self.run = Run(repo=repo, experiment=experiment_name, **kwargs)
        self._output_dir = output_dir

    def log_params(self, params: dict):
        for key, value in params.items():
            self.run[("params", key)] = value

    def log_metrics(self, metrics: dict, step: int = None):
        for key, value in metrics.items():
            self.run.track(value, name=key, step=step)

    def log_artifact(self, filepath: str):
        self.run.track_artifact(filepath)

    def finish(self):
        self.run.close()


def get_logger(logger_type: str = "file", **kwargs) -> ExperimentLogger:
    loggers = {
        "file": FileLogger,
        "comet": CometLogger,
        "aim": AimLogger,
    }
    logger_cls = loggers.get(logger_type.lower())
    if logger_cls is None:
        raise ValueError(f"Unknown logger type: {logger_type}. Choose from: {list(loggers.keys())}")
    return logger_cls(**kwargs)