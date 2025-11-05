from dataclasses import dataclass, field
from typing import Dict, List
import os

from torch.utils.tensorboard import SummaryWriter
import wandb
import torch
import numpy as np


@dataclass
class TrainingInfo:
    iter: int = 0
    elapsed_time: float = 0
    loss: int = None
    loss_dict: Dict[str, float] = field(default_factory=dict)
    densification_info: Dict[str, any] = field(default_factory=dict)
    eval_metrics: Dict[str, np.ndarray] = field(default_factory=dict)
    eval_thresholds: List[str] = field(default_factory=list)
    composed_image: np.ndarray = None


def get_logger(config: dict):
    if config.logger == "wandb":
        return WandbLogger(config)
    elif config.logger == "tensorboard":
        return TensorBoardLogger(config)
    else:
        return Logger(config)


class Logger:
    def __init__(self, config: dict):
        self.config = config
        self.experiment_name = os.path.split(config.model_path)[-1]

    def __del__(self):
        pass

    def log(self, training_info: TrainingInfo, gaussians):
        self.post_log(training_info)

    def post_log(self, training_info: TrainingInfo):
        training_info.loss_dict = {}
        training_info.densification_info = {}
        training_info.eval_metrics = {}
        training_info.eval_thresholds = []
        training_info.composed_image = None

    def compute_log_dict(self, training_info: TrainingInfo, gaussians):
        from scene import HairGaussianModel

        # Common data
        log_dict = {
            "general/iter_time": training_info.elapsed_time,
            "general/total_gaussians": (gaussians.get_xyz.shape[0]),
        }

        # HairGS specific
        if isinstance(gaussians, HairGaussianModel):
            log_dict["general/num_segments"] = gaussians.endpoint_pairs.shape[0]
            strands_info = gaussians.strands_info
            if strands_info is not None:
                total_strands = len(strands_info.list_strands)
                flatten_strands = np.concatenate(strands_info.list_strands)
                avg_strand_points = flatten_strands.shape[0] / total_strands
                log_dict["general/num_strands"] = total_strands
                log_dict["general/num_avg_strand_joints"] = avg_strand_points
                segments = gaussians._endpoints[gaussians.endpoint_pairs]
                avg_segment_length = (
                    torch.norm(segments[:, 0] - segments[:, 1], dim=1).mean().item()
                )
                log_dict["general/avg_segment_length"] = avg_segment_length
                log_dict["general/avg_strand_length"] = (
                    avg_strand_points * avg_segment_length
                )

        # Training losses
        if training_info.loss is not None:
            log_dict["train/loss"] = training_info.loss
        for k, v in training_info.loss_dict.items():
            log_dict[f"train/{k}"] = v

        # Densification
        for k, v in training_info.densification_info.items():
            log_dict[f"densification/{k}"] = v

        # Eval
        for metric, val_array in training_info.eval_metrics.items():
            if len(training_info.eval_thresholds) == 0:
                log_dict[f"eval/{metric}"] = val_array.mean()
            else:
                for threshold, val in zip(training_info.eval_thresholds, val_array):
                    log_dict[f"eval/{metric}@{threshold}"] = val
        return log_dict


class WandbLogger(Logger):
    def __init__(self, config: dict):
        super().__init__(config)
        wandb.login()
        log_config = {
            key: getattr(config, key) for key in vars(config) if not key.startswith("_")
        }
        self.run = wandb.init(
            project="HairGS",
            name=self.experiment_name,
            config=log_config,
        )

    def __del__(self):
        self.run.finish()

    def log(self, training_info: TrainingInfo, gaussians):
        log_dict = self.compute_log_dict(training_info, gaussians)
        if training_info.composed_image is not None:
            log_dict[f"images/{training_info.iter}-composed"] = wandb.Image(
                training_info.composed_image
            )
        wandb.log(log_dict, step=training_info.iter)
        self.post_log(training_info)


class TensorBoardLogger(Logger):
    def __init__(self, config: dict):
        super().__init__(config)
        self.writer = SummaryWriter(
            log_dir="./tensorboard_logs",
            comment=self.experiment_name,
        )

    def __del__(self):
        self.writer.close()

    def log(self, training_info: TrainingInfo, gaussians):
        log_dict = self.compute_log_dict(training_info, gaussians)
        for key, value in log_dict.items():
            self.writer.add_scalar(key, value, training_info.iter)
        if training_info.composed_image is not None:
            self.writer.add_image(
                f"images/{training_info.iter}-composed",
                training_info.composed_image,
                training_info.iter,
                dataformats="HWC",
            )
        self.post_log(training_info)
