"""Script for running inference and evaluation."""

import os
import time
import numpy as np
import hydra
import torch
import pandas as pd
import GPUtil
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist

from motifflow.experiments import utils as eu
from motifflow.models.structure.flow_module import FlowModule
from motifflow.data.datasets import PdbDataset

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)

class EvalRunner:
    def __init__(self, cfg: DictConfig):

        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))
        self._original_cfg = cfg.copy()

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up output directory only on rank 0
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            inference_dir = self.setup_inference_dir(ckpt_path)
            self._exp_cfg.inference_dir = inference_dir
            config_path = os.path.join(inference_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
            # dataset_cfg=eu.get_dataset_cfg(cfg),
            folding_cfg=self._infer_cfg.folding,
        )

        log.info(pl.utilities.model_summary.ModelSummary(self._flow_module))
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg

    @property
    def inference_dir(self):
        return self._flow_module.inference_dir
    
    def setup_inference_dir(self, ckpt_path):
        output_dir = os.path.join(
            self._infer_cfg.predict_dir,
            self._infer_cfg.task,
        )
        os.makedirs(output_dir, exist_ok=True)
        log.info(f'Saving results to {output_dir}')
        return output_dir

    def run_sampling(self):
        devices = GPUtil.getAvailable(order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        log.info(f'Evaluating {self._infer_cfg.task}')
        
        if self._infer_cfg.task == 'test_set':
            eval_dataset, _ = eu.dataset_creation(PdbDataset, self._cfg.pdb_post2021_dataset, 'inpainting')
        elif self._infer_cfg.task == 'unconditional':
            # eval_dataset = eu.LengthDataset(self._samples_cfg)
            raise ValueError(f'Not implemented task {self._infer_cfg.task}')
        elif self._infer_cfg.task == 'scaffolding':
            # eval_dataset = eu.ScaffoldingDataset(self._samples_cfg)
            raise ValueError(f'Not implemented task {self._infer_cfg.task}')
        else:
            raise ValueError(f'Unknown task {self._infer_cfg.task}')
        
        dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=devices)
        trainer.predict(self._flow_module, dataloaders=dataloader)

    def compute_designable_samples(self, output_dir):

        log.info(f'Calculating metrics for {output_dir}')
        top_sample_csv = eu.get_all_top_samples(output_dir)

        if self._infer_cfg.folding.use_pae:
            use_pae_condition = (top_sample_csv.mean_pae <= 5)
        else:
            use_pae_condition = True

        if self._infer_cfg.folding.self_consistency_metric == 'scRMSD':
            if self._infer_cfg.task == 'test_set':
                top_sample_csv['designable'] = (top_sample_csv.bb_rmsd_fold_sample <= 2.0) & (top_sample_csv.motif_bb_rmsd_fold_gt <= 1.0) & (top_sample_csv.mean_plddt >= 70) & use_pae_condition
            else: 
                raise ValueError(f'Unknown task {self._infer_cfg.task}')
            
        elif self._infer_cfg.folding.self_consistency_metric == 'scTM':
            if self._infer_cfg.task == 'test_set':
                top_sample_csv['designable'] = (top_sample_csv.tm_score_fold_sample >= 0.5) & (top_sample_csv.motif_bb_rmsd_fold_gt <= 1.0) & (top_sample_csv.mean_plddt >= 70) & use_pae_condition
            else: 
                raise ValueError(f'Unknown task {self._infer_cfg.task}')
        else:
            raise ValueError(f'Unknown top self-consistency scoring {self._infer_cfg.folding.self_consistency_metric}')

        metrics_df = pd.DataFrame(data={ 
            'Total designable': top_sample_csv.designable.sum(),
            'Designable': top_sample_csv.designable.mean(),
            'Total samples': len(top_sample_csv),
        }, index=[0])

        designable_csv_path = os.path.join(output_dir, 'designable.csv')
        metrics_df.to_csv(designable_csv_path, index=False)
        eu.calculate_diversity(output_dir, metrics_df, top_sample_csv, designable_csv_path, self._infer_cfg.folding.calculate_novelty)


@hydra.main(version_base=None, config_path="../configs", config_name="inference_joint_model")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_sampling()

    def compute_metrics():
        if cfg.inference.task == 'test_set':
            for subdir in os.scandir(sampler.inference_dir):
                if subdir.is_dir():
                    log.info(f'Computing metrics for subdirectory: {subdir.path}')
                    sampler.compute_designable_samples(subdir.path)
        else:
            raise ValueError(f'Unknown task {cfg.inference.task}')
    
    if cfg.inference.test_self_consistency:
        if dist.is_initialized():
            if dist.get_rank() == 0:
                compute_metrics()
        else:
            compute_metrics()

    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()