# Standard library imports
import os
import random
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
import matplotlib.pyplot as plt

from motifflow.models.base_module import BaseModule
from motifflow.models.latent.utils import compute_contact_map, compute_vae_loss, compute_vae_auc
from motifflow.models.structure.flow_model import FlowModel
from motifflow.models.structure import folding_model
from motifflow.models.structure.utils import compute_sample_metrics
from motifflow.models.structure import utils as msu
from motifflow.data.interpolant import Interpolant
from motifflow.data import all_atom, so3_utils, residue_constants, utils as du
from motifflow.analysis import metrics, utils as au
from motifflow.experiments import utils as eu
import shutil
from motifflow.models.structure.utils import run_pmpnn_post_processing
from biotite.sequence.io import fasta
# from openfold.utils.loss import between_residue_bond_loss_motif, calc_bb_fape


class FlowModule(BaseModule):
    
    def __init__(self, cfg, folding_cfg=None, folding_device_id=None):
        super().__init__(cfg)
        
        # Set up VAE
        if self._exp_cfg.load_ckpt:
            # Load weights from checkpoint
            state_dict = torch.load(self._exp_cfg.load_ckpt)['state_dict']
            encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()
                    if k.startswith('encoder.')}
            decoder_state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items()
                    if k.startswith('decoder.')}
            self.encoder.load_state_dict(encoder_state_dict)
            self.decoder.load_state_dict(decoder_state_dict)
            self._print_logger.info(f"Loaded encoder and decoder weights from {self._exp_cfg.load_ckpt}")
        else:
            self._print_logger.info("No VAE checkpoint provided")
            
        self.start_training_VAE_epoch = cfg.experiment.start_training_VAE_epoch
        self._interpolant_cfg = cfg.interpolant
            
        # Initialize model and interpolant
        self.structure_model = FlowModel(cfg.structure_model)
        self.interpolant = Interpolant(cfg.interpolant)

        self._folding_model = None
        self._folding_cfg = folding_cfg
        self._folding_device_id = folding_device_id

    @property
    def folding_model(self):
        if self._folding_model is None:
            self._folding_model = folding_model.FoldingModel(
                self._folding_cfg,
                device_id=self._folding_device_id
            )
        return self._folding_model
    
    def configure_optimizers(self):
        # Configure optimizer for training
        params = list(self.structure_model.parameters()) \
            + list(self.encoder.parameters()) \
            + list(self.decoder.parameters())
        return torch.optim.AdamW(params=params, **self._exp_cfg.optimizer)
    

    def loss_calculation(self, noisy_batch: Any, model_output: Any, z: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor):
        training_cfg = self._exp_cfg.training        

        # VAE loss
        recon_loss, kl_loss = compute_vae_loss(
            noisy_batch['pred_contact_logits'],
            compute_contact_map(noisy_batch['trans_1'], noisy_batch['residue_mask']),
            mean,
            logvar,
            motif_mask=noisy_batch['residue_mask']
        )
        vae_loss = (recon_loss + training_cfg.beta_max * kl_loss)

        # Structure loss
        loss_mask = noisy_batch['residue_mask'] # [B, N]
        
        if training_cfg.mask_plddt:
            loss_mask = loss_mask * noisy_batch['plddt_mask']
        else:
            loss_mask = loss_mask
        
        # Compute masks
        motif_region_condition_mask = loss_mask * noisy_batch['motif_mask']
        scaffold_region_infill_mask = loss_mask * noisy_batch['scaffold_mask']

        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch (loss_mask) encountered')
        #  if torch.any(torch.sum(motif_region_condition_mask, dim=-1) < 1):
        #      raise ValueError('Empty batch (motif_mask) encountered')
        #  if torch.any(torch.sum(scaffold_region_infill_mask, dim=-1) < 1):
        #      raise ValueError('Empty batch (scaffold_mask) encountered')

        num_batch, num_res = loss_mask.shape

        # Calculate ground truth values
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]

        # Timestep used for normalization.
        r3_t = noisy_batch['r3_t']
        so3_t = noisy_batch['so3_t']
        r3_norm_scale = 1 - torch.min(r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        so3_norm_scale = 1 - torch.min(so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))

        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError('NaN encountered in pred_rots_vf')
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]

        # loss Calcualte Function
        def calculate_loss(mask, gt, pred, norm_scale, add_scale, weight, max_value=None):
            error = (gt - pred) / norm_scale * add_scale
            loss_denom = torch.sum(mask, dim=-1) * 3
            loss_denom = torch.clamp(loss_denom, min=1e-8)
            loss = weight * torch.sum(error ** 2 * mask[..., None], dim=(-1, -2)) / loss_denom
            if max_value:
                loss = torch.clamp(loss, max=max_value)
            return loss

        # Translation VF loss
        all_trans_loss = calculate_loss(loss_mask, gt_trans_1, pred_trans_1, r3_norm_scale, training_cfg.trans_scale, training_cfg.translation_loss_weight, 5)
        motif_trans_loss = calculate_loss(motif_region_condition_mask, gt_trans_1, pred_trans_1, r3_norm_scale, training_cfg.trans_scale, training_cfg.translation_loss_weight, 5)
        scaffold_trans_loss = calculate_loss(scaffold_region_infill_mask, gt_trans_1, pred_trans_1, r3_norm_scale, training_cfg.trans_scale, training_cfg.translation_loss_weight, 5)

        # Rotation VF loss
        all_rots_vf_loss = calculate_loss(loss_mask, gt_rot_vf, pred_rots_vf, so3_norm_scale, 1, training_cfg.rotation_loss_weights, 5)
        motif_rots_vf_loss = calculate_loss(motif_region_condition_mask, gt_rot_vf, pred_rots_vf, so3_norm_scale, 1, training_cfg.rotation_loss_weights, 5)
        scaffold_rots_vf_loss = calculate_loss(scaffold_region_infill_mask, gt_rot_vf, pred_rots_vf, so3_norm_scale, 1, training_cfg.rotation_loss_weights, 5)

        # Backbone atom loss (aux)
        gt_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        
        def calculate_bb_atom_loss(mask, gt_bb, pred_bb, max_value=None):
            loss_denom = torch.sum(mask, dim=-1) * 3
            loss_denom = torch.clamp(loss_denom, min=1e-8)
            loss = torch.sum((gt_bb - pred_bb) ** 2 * mask[..., None, None], dim=(-1, -2, -3)) / loss_denom
            if max_value:
                loss = torch.clamp(loss, max=max_value)
            return loss

        all_bb_atom_loss = calculate_bb_atom_loss(loss_mask, gt_bb_atoms, pred_bb_atoms)
        motif_bb_atom_loss = calculate_bb_atom_loss(motif_region_condition_mask, gt_bb_atoms, pred_bb_atoms)
        scaffold_bb_atom_loss = calculate_bb_atom_loss(scaffold_region_infill_mask, gt_bb_atoms, pred_bb_atoms)

        # Pairwise distance loss (aux)
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)
            
        def calculate_pair_dist_loss(mask, gt_pair_dists, pred_pair_dists, max_value=None):
            # Expand loss mask from 2D to 3D and flatten (N,Ca,C)
            flat_loss_mask = torch.tile(mask[:, :, None], (1, 1, 3)).reshape([num_batch, num_res*3]) # shape: [B,N] -> [B, N*3]
            # Apply mask to distance matrices
            gt_pair_dists_masked = gt_pair_dists * flat_loss_mask[..., None]
            pred_pair_dists_masked = pred_pair_dists * flat_loss_mask[..., None]
            pair_dist_mask = flat_loss_mask[..., None] * flat_loss_mask[:, None, :]
            # Calculate distance matrix loss
            dist_mat_loss = torch.sum((gt_pair_dists_masked - pred_pair_dists_masked)**2 * pair_dist_mask, dim=(1, 2))
            mask_sum = torch.sum(pair_dist_mask, dim=(1, 2))
            # Set loss to 0 if mask sum is 0
            dist_mat_loss = torch.where(mask_sum > 0, 
                                      dist_mat_loss / mask_sum,
                                      torch.zeros_like(dist_mat_loss))
            if max_value:
                dist_mat_loss = torch.clamp(dist_mat_loss, max=max_value)

            return dist_mat_loss

        all_dist_mat_loss = calculate_pair_dist_loss(loss_mask, gt_pair_dists, pred_pair_dists)
        motif_dist_mat_loss = calculate_pair_dist_loss(motif_region_condition_mask, gt_pair_dists, pred_pair_dists)
        scaffold_dist_mat_loss = calculate_pair_dist_loss(scaffold_region_infill_mask, gt_pair_dists, pred_pair_dists)

        # Auxiliary loss Summation
        def calculate_auxiliary_loss(bb_atom_loss, dist_mat_loss, max_value=5):
            auxiliary_loss = (bb_atom_loss * training_cfg.aux_loss_use_bb_loss + dist_mat_loss * training_cfg.aux_loss_use_pair_loss)
            auxiliary_loss *= ((r3_t[:, 0] > training_cfg.aux_loss_t_pass) & (so3_t[:, 0] > training_cfg.aux_loss_t_pass))
            auxiliary_loss *= training_cfg.aux_loss_weight
            return torch.clamp(auxiliary_loss, max=max_value)

        all_auxiliary_loss = calculate_auxiliary_loss(all_bb_atom_loss, all_dist_mat_loss) # + contact_density_loss
        motif_auxiliary_loss = calculate_auxiliary_loss(motif_bb_atom_loss, motif_dist_mat_loss)
        scaffold_auxiliary_loss = calculate_auxiliary_loss(scaffold_bb_atom_loss, scaffold_dist_mat_loss)

        """
        # Bond length loss
        '''
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
        '''
        gt_atom37 = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :5]
        pred_atom37 = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)
        atom37_mask_motif = all_atom.to_atom37_mask(gt_trans_1, gt_rotmats_1)
        this_c_gt = gt_atom37[..., :-1, 2, :]
        next_n_gt = gt_atom37[..., 1:, 0, :]
        bond_loss = between_residue_bond_loss_motif(pred_atom37, atom37_mask_motif.to(pred_atom37.device), this_c_gt, next_n_gt)
        bond_loss['c_n_loss_mean'] *= self._exp_conf.c_n_loss_weight
        bond_loss['c_n_loss_mean'] *= (noisy_batch['so3_t'] < self._exp_conf.bb_atom_loss_t_filter)
        bond_loss['ca_c_n_loss_mean'] *= (noisy_batch['so3_t'] < self._exp_conf.bb_atom_loss_t_filter)
        bond_loss['c_n_ca_loss_mean'] *= (noisy_batch['so3_t'] < self._exp_conf.bb_atom_loss_t_filter)

        # motif FAPE loss
        bb_fape = calc_bb_fape(gt_rigids, pred_rigids, gt_atom37, pred_atom37, bb_mask)
        bb_fape *= (noisy_batch['so3_t'] < self._exp_conf.bb_atom_loss_t_filter)
        bb_fape *= self._exp_conf.bb_fape_loss_weight
        bb_fape *= self._exp_conf.aux_loss_weight
        """

        # Combine losses
        all_se3_vf_loss = all_trans_loss + all_rots_vf_loss # + all_auxiliary_loss
        motif_se3_vf_loss = motif_trans_loss + motif_rots_vf_loss # + motif_auxiliary_loss
        scaffold_se3_vf_loss = scaffold_trans_loss + scaffold_rots_vf_loss # + scaffold_auxiliary_loss

        # Weight structure loss
        weighted_se3_vf_loss = (training_cfg.motif_condition_loss_weight * motif_se3_vf_loss + scaffold_se3_vf_loss) + all_auxiliary_loss

        if torch.any(torch.isnan(weighted_se3_vf_loss)):
            raise ValueError('NaN loss encountered')
        
        # Final Loss
        total_loss = training_cfg.vae_loss_weight * vae_loss + weighted_se3_vf_loss

        return {
            "total_loss": total_loss,
            "vae_loss": vae_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "weighted_se3_vf_loss": weighted_se3_vf_loss,
            "trans_loss": all_trans_loss,
            "auxiliary_loss": all_auxiliary_loss,
            "rots_vf_loss": all_rots_vf_loss,
            "se3_vf_loss": all_se3_vf_loss,
            "motif_se3_vf_loss": motif_se3_vf_loss,
            "motif_trans_loss": motif_trans_loss,
            "motif_rots_vf_loss": motif_rots_vf_loss,
            "motif_auxiliary_loss": motif_auxiliary_loss,
            "scaffold_se3_vf_loss": scaffold_se3_vf_loss,
            "scaffold_trans_loss": scaffold_trans_loss,
            "scaffold_rots_vf_loss": scaffold_rots_vf_loss,
            "scaffold_auxiliary_loss": scaffold_auxiliary_loss,
        }
    
    def on_train_epoch_start(self):
        if self.current_epoch < self.start_training_VAE_epoch:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        elif self.current_epoch == self.start_training_VAE_epoch:
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Perform a single training step
        step_start_time = time.time()
        self.interpolant.set_device(batch['residue_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        
        # VAE and latent diffusion
        noisy_batch['group_mask'] = noisy_batch['motif_mask']   # Add group mask
        z_all, mean_all, logvar_all = self.encoder(
            batch['trans_1'], batch['rotmats_1'], batch['aatype'],
            batch['residue_mask'], batch['residue_mask'],
            batch["residue_index"], batch["chain_index"]
        )
        pred_contact_map, pred_contact_logits = self.decoder(z_all, batch['residue_mask'], batch['residue_mask'])
        
        noisy_batch['z_all'] = z_all
        noisy_batch['pred_contact_map'] = pred_contact_map
        noisy_batch['pred_contact_logits'] = pred_contact_logits
        
        # Apply self-conditioning if enabled
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.structure_model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans'] # * noisy_batch['scaffold_mask'][..., None]
                noisy_batch['rotmats_sc'] = model_sc['pred_rotmats'] # * noisy_batch['scaffold_mask'][..., None, None]
        
        model_output = self.structure_model(noisy_batch)

        # Calculate losses and log metrics
        batch_losses = self.loss_calculation(noisy_batch, model_output, z_all, mean_all, logvar_all)

        # Log various training metrics
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {k: torch.mean(v) for k,v in batch_losses.items()}
        for k, v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # log mean and varaiance
        roc_auc, pr_auc = compute_vae_auc(
            noisy_batch['pred_contact_map'],
            compute_contact_map(noisy_batch['trans_1'], noisy_batch['residue_mask']),
            motif_mask=noisy_batch['residue_mask']
        )
        self._log_scalar("train/mean", np.mean(du.to_numpy(mean_all)), prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/logvar", np.mean(du.to_numpy(logvar_all)), prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/roc_auc", np.mean(roc_auc), prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/pr_auc", np.mean(pr_auc), prog_bar=False, batch_size=num_batch)
        
        # Log time step metrics
        so3_t = torch.squeeze(noisy_batch['so3_t'])
        r3_t = torch.squeeze(noisy_batch['r3_t'])
        self._log_scalar("train/so3_t", np.mean(du.to_numpy(so3_t)), prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/r3_t", np.mean(du.to_numpy(r3_t)), prog_bar=False, batch_size=num_batch)

        # Log stratified losses across time steps
        excluded_losses = {'vae_loss', 'recon_loss', 'kl_loss'}
        filtered_losses = {k: v for k, v in batch_losses.items() if k not in excluded_losses}
        for loss_name, loss_dict in filtered_losses.items():
            if loss_name == 'rots_vf_loss' or 'motif_rots_vf_loss' or 'scaffold_rots_vf_loss':
                batch_t = so3_t
            else:
                batch_t = r3_t
            stratified_losses = msu.t_stratified_loss(batch_t, loss_dict, loss_name=loss_name)
            for k, v in stratified_losses.items():
                self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Calculate and log training throughput
        # Log throughput and related metrics
        scaffold_percent = torch.mean(batch['scaffold_mask'].float()).item()
        self._log_scalar("train/scaffolding_percent", scaffold_percent, prog_bar=False, batch_size=num_batch)
        
        num_motif_res = torch.sum(batch['motif_mask'].float() * (batch['chain_index'] == 1).float(), dim=-1)
        self._log_scalar("train/motif_size", torch.mean(num_motif_res).item(), prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/length", batch['residue_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)

        step_time = time.time() - step_start_time
        self._log_scalar("train/examples_per_second", num_batch / step_time)

        train_loss = total_losses['total_loss']
        # self._log_scalar("train/weighted_loss", train_loss, batch_size=num_batch)
        # self._log_scalar("train/ori_loss", total_losses['se3_vf_loss'], batch_size=num_batch)

        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.structure_model.parameters(), max_norm=1.0)
        
        return train_loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # Perform a single validation step
        res_mask = batch['residue_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        csv_index = batch['csv_idx']
        true_contacts = compute_contact_map(batch['trans_1'], batch['residue_mask'])

        z_all, mean, logvar = self.encoder(
            batch['trans_1'], batch['rotmats_1'], batch['aatype'],
            batch['residue_mask'], batch['residue_mask'],
            batch["residue_index"], batch["chain_index"]
        )
        pred_contact_map, pred_contact_logits = self.decoder(z_all, batch['residue_mask'], batch['residue_mask'])
        roc_auc, pr_auc = compute_vae_auc(
            pred_contact_map,
            true_contacts,
            batch['residue_mask']
        )

        VAE_metrics = {
            'mean': mean,
            'logvar': logvar,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
        }

        # log mean and varaiance
        self._log_scalar("valid/mean", np.mean(du.to_numpy(mean)), on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch)
        self._log_scalar("valid/logvar", np.mean(du.to_numpy(logvar)), on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch)
        self._log_scalar("valid/roc_auc", roc_auc, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch)
        self._log_scalar("valid/pr_auc", pr_auc, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch)
        
        # Generate samples
        atom37_traj, _, _, _ = self.interpolant.sample(
            num_batch, num_res, self.structure_model, residue_mask=batch['residue_mask'],
            trans_1=batch['trans_1'], rotmats_1=batch['rotmats_1'], 
            scaffold_mask=batch['scaffold_mask'], motif_mask=batch['motif_mask'],
            chain_index=batch['chain_index'], residue_index=batch['residue_index'], 
            fixed_structure_mask=batch['fixed_structure_mask'], aatype=batch['aatype'], group_mask=batch['motif_mask'],
            z_all=z_all, pred_contact_map=pred_contact_map
        )
        samples = atom37_traj[-1].numpy()
        assert samples.shape == (num_batch, num_res, 37, 3)

        # Process each sample in the batch
        batch_metrics = []
        true_bb_pos = all_atom.atom37_from_trans_rot(batch['trans_1'], batch['rotmats_1'])
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f'sample_{csv_index[i].item()}_{batch["pdb_name"][i]}_idx_{batch_idx}_len_{num_res}'
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Save ground truth
            au.write_prot_to_pdb(
                prot_pos=true_bb_pos[i].cpu().detach().numpy(),
                file_path=os.path.join(sample_dir, 'sample_gt.pdb'),
                aatype=du.to_numpy(batch['aatype'] * batch['residue_mask'].long())[i],
                chain_index=batch['chain_index'][i],
                no_indexing=True
            )
            
            # Save sample as PDB file and log to W&B if applicable
            final_pos = samples[i]
            motif_aatype = du.to_numpy(batch['aatype'] * batch['motif_mask'].long())[i]
            # Save sample
            saved_path = au.write_prot_to_pdb(
                final_pos, 
                os.path.join(sample_dir, 'sample.pdb'), 
                aatype=motif_aatype, 
                chain_index=batch['chain_index'][i],
                no_indexing=True)
            
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append([saved_path, self.global_step, wandb.Molecule(saved_path)])
            
            # Run designability
            if self._exp_cfg.training.valid_designability:
                pmpnn_pdb_path = saved_path.replace('.pdb', '_pmpnn.pdb')
                shutil.copy(saved_path, pmpnn_pdb_path)
                pmpnn_fasta_path = run_pmpnn_post_processing(self.folding_model, sample_dir, pmpnn_pdb_path, batch['motif_mask'][i], batch['chain_index'][i])
                folded_dir = os.path.join(sample_dir, 'folded')
                os.makedirs(folded_dir, exist_ok=True)
                folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)
                true_bb_pos = all_atom.atom37_from_trans_rot(batch['trans_1'][i].unsqueeze(0), batch['rotmats_1'][i].unsqueeze(0), batch['motif_mask'][i].unsqueeze(0))
                true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
                designable_results = msu.process_folded_outputs(
                    saved_path, folded_output,
                    true_bb_pos,
                    batch['motif_mask'][i],
                    batch['motif_mask'][i],
                    batch['chain_index'][i]) 
            
                designable_metrics = {
                    'bb_rmsd': designable_results.bb_rmsd_fold_sample.min(),
                    'motif_bb_rmsd': designable_results.motif_bb_rmsd_fold_sample.min(),
                    'motif_rmsd': designable_results.motif_rmsd_fold_sample.min(),
                }
            else:
                designable_metrics = {}

            try:
                mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
                ca_idx = residue_constants.atom_order['CA']
                ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
                batch_metrics.append((mdtraj_metrics | ca_ca_metrics | designable_metrics))
            except Exception as e:
                print(e)
                continue

            # plot true contact map and predicted contact map
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))
            # True contact map
            true_contact_map = true_contacts[i].cpu().detach().numpy()
            ax1.imshow(true_contact_map)
            ax1.set_title('True Contact Map')
            # decoded contact map
            decoded_contact_map = pred_contact_map[i].cpu().detach().numpy()
            ax2.imshow(decoded_contact_map)
            ax2.set_title('Decoded Contact Map')
            # Predicted contact map  
            ca_positions = torch.from_numpy(final_pos[:, ca_idx]).unsqueeze(0).to(res_mask.device)
            recon_contact_map = compute_contact_map(ca_positions, batch['residue_mask'][i].unsqueeze(0)).squeeze(0).cpu().detach().numpy()
            ax3.imshow(recon_contact_map)
            ax3.set_title('Predicted Contact Map')
            # Contact map difference
            contact_map_diff = np.abs(true_contact_map - recon_contact_map)
            ax4.imshow(contact_map_diff)
            ax4.set_title('Contact Map Difference between True and Predicted')
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'contact_maps.png'))
            plt.close()

            # Cacluate fodling mettics
            true_ca_pos = true_bb_pos[i][:, ca_idx, :].unsqueeze(0)  # (1, N, 3)
            lddt = metrics.cal_atom_lddt(ca_positions, true_ca_pos, batch['residue_mask'][i].unsqueeze(0))
            # _, tm_score = metrics.calc_tm_score(ca_positions, true_ca_pos, du.aatype_to_seq(batch['aatype'][i]), du.aatype_to_seq(batch['aatype'][i]))
            rmsd = metrics.cal_aligned_rmsd_2(ca_positions.squeeze(0), true_ca_pos.squeeze(0))
            prediction_metrics = {
                'lddt': lddt,
                # 'tm_score': tm_score,
                'rmsd': rmsd,
            }
            batch_metrics.append(prediction_metrics)

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)

    def on_validation_epoch_end(self):
        # Log validation metrics at the end of each epoch
        if len(self.validation_epoch_samples) > 0 :
            self.logger.log_table(key='valid/samples', columns=["sample_path", "global_step", "Protein"], data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        super().on_validation_epoch_end()
    
    @torch.no_grad()
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Perform prediction step

        In Unconditional Task
        In Inpainting (scaffolding) Task
        In test_set Task
            'trans_1': Translation vectors (motif-centered)
            'rotmats_1': Rotation matrices
            'residue_index': Residue indices
            'chain_index': Chain indices
            'residue_mask': Residue mask
            'ressidue_plddt': Per-residue pLDDT scores
            'plddt_mask': pLDDT mask (all 1 if no use)
            'aatype': Amino acid types as int
            'scaffold_mask': scaffolding mask
            'fixed_structure_mask'
            'csv_idx': CSV index
            'pdb_name': PDB name
        """
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant)
        interpolant.set_device(device)

        if 'sample_id' in batch:
            sample_ids = batch['sample_id'].squeeze().tolist()
        else:
            sample_ids = [0]
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)

        if self._infer_cfg.task == 'test_set': # motif-scaffolding
            _, sample_length, _ = batch['trans_1'].shape
            sample_dirs = [os.path.join(self.inference_dir, f'length_{sample_length}', batch['pdb_name'][0])]
            for sample_dir in sample_dirs:
                os.makedirs(sample_dir, exist_ok=True)
            true_bb_pos = all_atom.atom37_from_trans_rot(batch['trans_1'], batch['rotmats_1'])
            assert true_bb_pos.shape == (1, sample_length, 37, 3)
            # save the ground truth as a pdb
            au.write_prot_to_pdb(
                prot_pos=true_bb_pos[0].cpu().detach().numpy(),
                file_path=os.path.join(sample_dirs[0], batch['pdb_name'][0] + '_gt.pdb'),
                aatype=batch['aatype'][0].cpu().detach().numpy(),
            )
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy() 
            assert true_bb_pos.shape == (sample_length * 3, 3)
            group_mask = batch['motif_mask']
        
        else:
            raise ValueError(f'Not implemented task {self._infer_cfg.task}')
        
        # Skip runs if already exist
        top_sample_csv_paths = [os.path.join(sample_dir, 'top_sample.csv')
                                for sample_dir in sample_dirs]
        if all([os.path.exists(top_sample_csv_path) for top_sample_csv_path in top_sample_csv_paths]):
            self._print_logger.info(f'Skipping instance {sample_ids} length {sample_length}')
            return
        

        # VAE
        z_all, mean, logvar = self.encoder(
            batch['trans_1'], batch['rotmats_1'], batch['aatype'],
            batch['residue_mask'], batch['residue_mask'],
            batch["residue_index"], batch["chain_index"]
        )
        pred_contact_map, pred_contact_logits = self.decoder(z_all, batch['residue_mask'], batch['residue_mask'])

        # Generate samples
        atom37_traj, model_traj, _, _ = interpolant.sample(
            num_batch, sample_length, self.structure_model, residue_mask=batch['residue_mask'],
            trans_1=batch['trans_1'], rotmats_1=batch['rotmats_1'], 
            scaffold_mask=batch['scaffold_mask'], motif_mask=batch['motif_mask'],
            chain_index=batch['chain_index'], residue_index=batch['residue_index'], 
            fixed_structure_mask=batch['fixed_structure_mask'], aatype=batch['aatype'], group_mask=group_mask,
            z_all=z_all, pred_contact_map=pred_contact_map
        )
        samples = atom37_traj[-1].numpy()
        assert samples.shape == (num_batch, sample_length, 37, 3)

        if self._infer_cfg.task == 'test_set':
            motif_true_bb_pos = all_atom.atom37_from_trans_rot(batch['trans_1'], batch['rotmats_1'], batch['motif_mask'])
            motif_true_bb_pos = motif_true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            assert motif_true_bb_pos.shape == (sample_length * 3, 3)

        # Process and save each sample
        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        noisy_traj_length = bb_trajs.shape[1]
        assert bb_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)
        model_trajs = du.to_numpy(torch.stack(model_traj, dim=0).transpose(0, 1))
        clean_traj_length = model_trajs.shape[1]
        assert model_trajs.shape == (num_batch, clean_traj_length, sample_length, 37, 3)
        
        for i, sample_id in zip(range(num_batch), sample_ids):
            sample_dir = sample_dirs[i]
            os.makedirs(sample_dir, exist_ok=True)

            if 'aatype' in batch: # motif scaffolding
                # aatype = du.to_numpy(batch['aatype'].long())[0]
                aatype = du.to_numpy(batch['aatype'] * batch['motif_mask'].long())[0]
            else: # Unconditional
                aatype = np.zeros(sample_length, dtype=int)

            # Test self_consistency
            if self._infer_cfg.test_self_consistency:
                top_sample_df = compute_sample_metrics(model_trajs[i], bb_trajs[i], motif_true_bb_pos, aatype, batch['motif_mask'], 
                                                            group_mask, batch['scaffold_mask'], batch['chain_index'], sample_id, sample_length, sample_dir,
                                                            self.folding_model, self._infer_cfg.folding.self_consistency_metric, self._infer_cfg.write_sample_trajectories)
                top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
                top_sample_df.to_csv(top_sample_csv_path)
            else: 
                _ = eu.save_traj(
                    bb_trajs[i][-1],
                    bb_trajs[i],
                    np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                    du.to_numpy(batch['scaffold_mask'])[0],
                    chain_index=batch['chain_index'][0],
                    output_dir=sample_dir,
                    aatype=aatype,)

            # plot true contact map and predicted contact map
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))
            # True contact map
            true_contacts = compute_contact_map(batch['trans_1'], batch['residue_mask'])
            true_contact_map = true_contacts[i].cpu().detach().numpy()
            ax1.imshow(true_contact_map)
            ax1.set_title('True Contact Map')
            # decoded contact map
            decoded_contact_map = pred_contact_map[i].cpu().detach().numpy()
            ax2.imshow(decoded_contact_map)
            ax2.set_title('Decoded Contact Map')
            # Predicted contact map  
            ca_idx = residue_constants.atom_order['CA']
            ca_positions = torch.from_numpy(samples[i][:, ca_idx]).unsqueeze(0).to(device)
            recon_contact_map = compute_contact_map(ca_positions, batch['residue_mask'][i].unsqueeze(0)).squeeze(0).cpu().detach().numpy()
            ax3.imshow(recon_contact_map)
            ax3.set_title('Predicted Contact Map')
            # Contact map difference
            contact_map_diff = np.abs(true_contact_map - recon_contact_map)
            ax4.imshow(contact_map_diff)
            ax4.set_title('Contact Map Difference between True and Predicted')
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'contact_maps.png'))
            plt.close()