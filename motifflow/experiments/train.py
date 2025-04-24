import os
import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from motifflow.data.datasets import ScopeDataset, PdbDataset, AFDBDataset
from motifflow.data.protein_dataloader import ProteinDataModule
from motifflow.models.latent.vae_module import ProteinVAEModule
# from motifflow.models.latent.latent_diffusion import ProteinLatentDiffusionModule
from motifflow.models.structure.flow_module import FlowModule
from motifflow.experiments import utils as eu
from motifflow.models.structure.lawa import LAWACallback

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

class Experiment:
    
    def __init__(self, cfg: DictConfig):

        self._cfg = cfg
        self._setup_dataset()

        self._datamodule: LightningDataModule = ProteinDataModule(
            data_cfg=self._cfg.data,
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset
        )
        if self._cfg.experiment.objective == 'VAE':
            self._module: LightningModule = ProteinVAEModule(self._cfg)
        elif self._cfg.experiment.objective == 'joint':
            total_devices = self._cfg.experiment.num_devices
            device_ids = eu.get_available_device(total_devices)
            self._train_device_ids = device_ids
            log.info(f"Training with devices: {self._train_device_ids}")
            self._module: LightningModule = FlowModule(
                self._cfg,
                folding_cfg=self._cfg.folding,
                folding_device_id=self._train_device_ids[0]
                )
        # elif self._cfg.experiment.objective == 'latent_diffusion':
            # self._module: LightningModule = ProteinLatentDiffusionModule(self._cfg)
        # elif self._cfg.experiment.objective == 'SDM':
        #     self._module: LightningModule = SDMModule(self._cfg)
        else:
            raise ValueError(f"Unknown training objective: {self._cfg.experiment.objective}")
        
        self._train_device_ids = eu.get_available_device(self._cfg.experiment.num_devices)
        log.info(f"Training with devices: {self._train_device_ids}")


    def _setup_dataset(self):
        if self._cfg.data.dataset == 'scope':
            dataset_class = ScopeDataset
        elif self._cfg.data.dataset == 'pdb':
            dataset_class = PdbDataset
        elif self._cfg.data.dataset == 'AFDB':
            dataset_class = AFDBDataset
        # Get configuration for the selected dataset
        dataset_cfg = getattr(self._cfg, f"{self._cfg.data.dataset}_dataset")
        # Create training and validation datasets
        self.train_dataset, self.valid_dataset = eu.dataset_creation(dataset_class, dataset_cfg, self._cfg.data.task)


    def train(self):

        callbacks = []
        if self._cfg.experiment.debug:
            log.info("Debug mode.")
            logger = None
            self._train_device_ids = [self._train_device_ids[0]]
            self._cfg.data.loader.num_workers = 0
        else:
            logger = WandbLogger(**self._cfg.experiment.wandb)
            ckpt_dir = self._cfg.experiment.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")
            callbacks.append(ModelCheckpoint(**self._cfg.experiment.checkpointer))

            if self._cfg.experiment.lawa.use_lawa:
                lawa_callback = LAWACallback(
                    k=self._cfg.experiment.lawa.lawa_k,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    use_ema=self._cfg.experiment.lawa.lawa_use_ema,
                    decay_rate=self._cfg.experiment.lawa.lawa_decay_rate,
                    lawa_start_epoch=self._cfg.experiment.lawa.lawa_start_epoch
                )
                callbacks.append(lawa_callback)
                log.info(f"LAWA callback added with decay rate {self._cfg.experiment.lawa.lawa_decay_rate} starting from epoch {self._cfg.experiment.lawa.lawa_start_epoch}")
                    
            local_rank = os.environ.get('LOCAL_RANK', 0)
            if local_rank == 0:
                cfg_path = os.path.join(ckpt_dir, 'config.yaml')
                with open(cfg_path, 'w') as f:
                    OmegaConf.save(config=self._cfg, f=f.name)
                cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
                flat_cfg = dict(eu.flatten_dict(cfg_dict))
                if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                    logger.experiment.config.update(flat_cfg)

        trainer = Trainer(
            **self._cfg.experiment.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids,
        )

        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._cfg.experiment.warm_start
        )


@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):
    
    if cfg.experiment.warm_start and cfg.experiment.warm_start_cfg_override:
        # load warm start
        warm_start_cfg_path = os.path.join(os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)
        # Unpack and merge model configurations
        model_configs = ['vae_model', 'latent_diffusion', 'structure_model']
        for model_config in model_configs:
            OmegaConf.set_struct(cfg[model_config], False)
            OmegaConf.set_struct(warm_start_cfg[model_config], False)
            cfg[model_config] = OmegaConf.merge(cfg[model_config], warm_start_cfg[model_config])
            OmegaConf.set_struct(cfg[model_config], True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg)
    exp.train()

if __name__ == "__main__":
    main()
