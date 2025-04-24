import torch
import torch.distributed as dist
import os
from typing import Dict
from pytorch_lightning import LightningModule
from motifflow.models.latent.vae_model import Encoder_Transformer, Encoder_MPNN, Decoder
import logging
import pandas as pd
import time

class BaseModule(LightningModule):
    def __init__(self, cfg: Dict):
        super().__init__()
        
        # Set up configuration
        self._cfg = cfg
        self._vae_model_cfg = cfg.vae_model
        self._exp_cfg = cfg.experiment
        self._training_cfg = cfg.experiment.training
        self._data_cfg = cfg.data

        # Initialize model
        if self._vae_model_cfg.encoder.type == "MPNN":
            self.encoder = Encoder_MPNN(self._vae_model_cfg.encoder.MPNN)
        elif self._vae_model_cfg.encoder.type == "Transformer":
            self.encoder = Encoder_Transformer(self._vae_model_cfg.encoder.Transformer)
        else:
            raise ValueError(f"Encoder type {self._vae_model_cfg.encoder.type} not supported")
        self.decoder = Decoder(self._vae_model_cfg.decoder)

        # Set up logging
        self._print_logger = logging.getLogger(__name__)
        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self._checkpoint_dir = None
        self._inference_dir = None

        self.save_hyperparameters()

    @property
    def checkpoint_dir(self):
        # Lazy initialization of checkpoint directory
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir
    
    def _log_scalar(self, key, value, on_step=True, on_epoch=False, prog_bar=True, batch_size=None, sync_dist=False, rank_zero_only=True):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(key, value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=batch_size, sync_dist=sync_dist, rank_zero_only=rank_zero_only)
    
    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()
    
    def on_validation_epoch_end(self):
        # Log validation metrics at the end of each epoch
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(f'valid/{metric_name}', metric_val, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(val_epoch_metrics))
        self.validation_epoch_metrics.clear()

    