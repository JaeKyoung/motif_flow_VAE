import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import pandas as pd
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from motifflow.models.base_module import BaseModule
from copy import deepcopy
from motifflow.modules.diffusion.create_diffusion import create_diffusion
from motifflow.models.structure.flow_module import FlowModel
from motifflow.data.interpolant import Interpolant
from motifflow.models.latent.latent_diffusion import ProteinLatentDiT
from collections import OrderedDict
from motifflow.models.latent.utils import compute_contact_map, compute_vae_auc, compute_vae_loss
from motifflow.data import all_atom, residue_constants
import matplotlib.pyplot as plt
import numpy as np
import os
from motifflow.models.structure import utils as msu
from motifflow.data import utils as du
import time
import shutil
from motifflow.analysis import metrics, utils as au
from motifflow.models.structure.utils import run_pmpnn_post_processing



#########################################################
# ProteinLatentDiffusionModule
#########################################################

@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999):
    """
    Performs Exponential Moving Average update on model parameters.

    Args:
        ema_model: The exponential moving average model.
        model: The current model.
        decay: The decay factor for EMA.
    """
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())

    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)


class ProteinLatentDiffusionModule(BaseModule):
    def __init__(self, cfg: Dict, folding_cfg=None, folding_device_id=None):
        super().__init__(cfg)
        self.structure_model = FlowModel(cfg.structure_model)
        self._interpolant_cfg = cfg.interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        # Set up VAE
        if self._exp_cfg.load_ckpt:
            # Load weights from checkpoint
            self._print_logger.info(f"Loading weights from {self._exp_cfg.load_ckpt}")
            state_dict = torch.load(self._exp_cfg.load_ckpt)['state_dict']
            encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()
                                if k.startswith('encoder.')}
            self.encoder.load_state_dict(encoder_state_dict)
            decoder_state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items()
                                if k.startswith('decoder.')}
            self.decoder.load_state_dict(decoder_state_dict)
            structure_model_state_dict = {k.replace('structure_model.', ''): v for k, v in state_dict.items()   
                                if k.startswith('structure_model.')}
            self.structure_model.load_state_dict(structure_model_state_dict)
            self._print_logger.info("Loaded VAE and structure model weights from checkpoint.")
        else:
            raise ValueError("No checkpoint provided. Please check the checkpoint path.")

        # Freeze parameters during latent diffusion training
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.structure_model.parameters():
            param.requires_grad = False 
        self._print_logger.info("VAE encoder, decoder, and structure model parameters frozen.")
        
        # Set up latent diffusion model
        self.latent_diffusion = ProteinLatentDiT(self._cfg.latent_diffusion)
        self.diffusion = create_diffusion(
            timestep_respacing="", 
            diffusion_steps=self._cfg.latent_diffusion.num_timesteps,
            learn_sigma=self._cfg.latent_diffusion.learn_sigma
        )
        
        # EMA Model Setup
        self.ema = deepcopy(self.latent_diffusion).to(self.device)
        for p in self.ema.parameters():
            p.requires_grad = False
        self.ema_decay = self._cfg.latent_diffusion.ema_decay

        self._folding_model = None
        self._folding_cfg = folding_cfg
        self._folding_device_id = folding_device_id
    
    def structure_loss(self, samples, batch):
        '''
        Calculate structure loss
        '''
        pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        '''
        Training latent diffusion model
        '''
        step_start_time = time.time()
        B, num_res, _ = batch['trans_1'].shape
        device = batch['trans_1'].device

        # Encode the all protein (gt)
        with torch.no_grad():
            z_0, mean, logvar = self.encoder.encode(
                batch['trans_1'], batch['rotmats_1'], batch['aatype'],
                batch['residue_mask'], batch['residue_mask'],
                batch["residue_index"], batch["chain_index"]
            )

        # Sample a random timestep
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device).long()
        noise = torch.randn_like(z_0)

        # Forward diffusion process
        z_t = self.diffusion.q_sample(x_start=z_0, t=t, noise=noise)  # Add noise to z_0
        # noised_contact_map, pred_contact_logits_denoised = self.decoder(z_t, batch['residue_mask'], batch['motif_mask'])
        # Predict the noise added at step t using the latent diffusion model
        eps_pred = self.latent_diffusion(z_t, t, batch['motif_mask']) # (B, N, D)
        loss_diffusion = F.mse_loss(eps_pred, noise)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1).mean()
        
        # Denoise the sample
        '''
        denoised_z = self.diffusion.p_sample_loop(
            self.latent_diffusion,
            z_0.shape,
            noise=z_t,  # Use the same noisy sample
            model_kwargs={'motif_mask': batch['motif_mask']},
            device=device,
            progress=False
        )
        
        # Decode denoised samples to get contact maps
        pred_contact_map_denoised, pred_contact_logits_denoised = self.decoder(denoised_z, batch['residue_mask'], batch['motif_mask'])
        batch["z_all"] = denoised_z
        batch['pred_contact_map'] = pred_contact_map_denoised
        batch['pred_contact_logits'] = pred_contact_logits_denoised

        true_contacts = compute_contact_map(batch['trans_1'], batch['residue_mask'])

        eps = 1e-8
        with torch.no_grad():
            alpha = true_contacts.mean().clamp(min=eps, max=1.0-eps)
            pos_weight = 1.0 / alpha
            neg_weight = 1.0 / (1.0 - alpha)
            weights = torch.where(true_contacts > 0.5, pos_weight, neg_weight)

        contact_map_loss = F.binary_cross_entropy_with_logits(
            pred_contact_logits_denoised,
            true_contacts,
            weight=weights,
            reduction='none'
        )
        outer_residue_mask = batch['residue_mask'].unsqueeze(1) * batch['residue_mask'].unsqueeze(2)
        contact_map_loss = (contact_map_loss * outer_residue_mask).sum() / (outer_residue_mask.sum() + eps)
        '''
        # TODO
        # Generate denoised samples for structure loss
        if self._exp_cfg.training.use_structure_loss:
            # Generate structure from denoised samples
            self.interpolant.set_device(device)
            atom37_traj, _, _, _ = self.interpolant.sample(
                B, num_res, self.structure_model, residue_mask=batch['residue_mask'],
                trans_1=batch['trans_1'], rotmats_1=batch['rotmats_1'], 
                scaffold_mask=batch['scaffold_mask'], motif_mask=batch['motif_mask'],
                chain_index=batch['chain_index'], residue_index=batch['residue_index'], 
                fixed_structure_mask=batch['fixed_structure_mask'], aatype=batch['aatype'], group_mask=batch['motif_mask'],
                z_all=batch["z_all"], pred_contact_map=batch["pred_contact_map"]
            )
            samples = atom37_traj[-1].numpy()
            assert samples.shape == (B, num_res, 37, 3)
            structure_loss = 0.0
        else:
            structure_loss = 0.0
        
        # Combine all losses
        total_loss = (
            loss_diffusion + 
            self._training_cfg.beta_max * kl_loss + 
            self._training_cfg.structure_loss_weight * structure_loss # + 
            # self._training_cfg.contact_map_loss_weight * contact_map_loss
        )

        losses = {
            'total_loss': total_loss,
            'loss_diffusion': loss_diffusion,
            'kl_loss': kl_loss,
            'structure_loss': structure_loss,
            # 'contact_map_loss': contact_map_loss
        }
        for k, v in losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=B)

        '''
        roc_auc, pr_auc = compute_vae_auc(
            batch['pred_contact_map'],
            compute_contact_map(batch['trans_1'], batch['residue_mask']),
            motif_mask=batch['residue_mask']
        )
        self._log_scalar("train/roc_auc", np.mean(roc_auc), prog_bar=False, batch_size=B)
        self._log_scalar("train/pr_auc", np.mean(pr_auc), prog_bar=False, batch_size=B)
        '''
        t = torch.squeeze(t)
        self._log_scalar("train/t", np.mean(du.to_numpy(t)), prog_bar=False, batch_size=B)

        excluded_losses = {}
        filtered_losses = {k: torch.tensor(v, device=device) if isinstance(v, (float, int)) else v 
                           for k, v in losses.items() if k not in excluded_losses}
        for loss_name, loss_dict in filtered_losses.items():
            batch_t = torch.clamp(t.float() / float(self.diffusion.num_timesteps), 0.0, 1.0)
            stratified_losses = msu.t_stratified_loss(batch_t, loss_dict, loss_name=loss_name)
            for k, v in stratified_losses.items():
                self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=B)
        
        self._log_scalar("train/batch_size", B, prog_bar=False)

        step_time = time.time() - step_start_time
        self._log_scalar("train/examples_per_second", B / step_time)

        return total_loss
    
    def on_train_epoch_end(self):
        # Update EMA model at the end of each epoch
        update_ema(self.ema, self.latent_diffusion, decay=self.ema_decay)
        super().on_train_epoch_end()
    
    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        '''
        Validation step for latent diffusion model
        '''
        batch_size, num_res, _ = batch['trans_1'].shape
        device = batch['trans_1'].device
        csv_index = batch['csv_idx']
        
        # VAE Encoding
        z_all, mean_all, logvar_all = self.encoder(
            batch['trans_1'], batch['rotmats_1'], batch['aatype'],
            batch['motif_mask'], batch['residue_mask'],
            batch["residue_index"], batch["chain_index"]
        )

        # --- Latent Diffusion Sampling ---
        # Use the EMA model for validation/inference for more stable results
        ema_model = self.ema
        shape = z_all.shape # Shape will be (B, N, latent_dim)

        # Generate samples starting from noise z_T
        z_T = torch.randn_like(z_all)
        samples_z = self.diffusion.p_sample_loop(
            ema_model,
            shape,
            noise=z_T,
            model_kwargs={'motif_mask': batch['motif_mask']}, # Pass necessary conditioning
            device=device,
            progress=False # Disable progress bar for validation steps
        ) # samples_z contains the final denoised latent z_0

        # --- VAE Decoding ---
        # Decode the sampled latent variables to get predicted contact maps
        pred_contact_map_sampled, pred_contact_logits_sampled = self.decoder(
            samples_z, batch['residue_mask'], batch['residue_mask']
        )

        # --- Loss Calculation (Optional but useful) ---
        # Calculate diffusion loss using the non-EMA model for consistency with training objective
        # Note: This requires sampling t and noise again, which might not be standard validation practice.
        # Alternatively, evaluate metrics on the *sampled* data.

        # Example: Calculate reconstruction quality (e.g., AUC for contact map)
        true_contacts = compute_contact_map(batch['trans_1'], batch['residue_mask']) #
        roc_auc, pr_auc = compute_vae_auc(pred_contact_map_sampled, true_contacts, batch['residue_mask']) #
        self._log_scalar("valid/roc_auc", roc_auc, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self._log_scalar("valid/pr_auc", pr_auc, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        # Log latent space stats (optional)
        self._log_scalar("valid/latent_mean_norm", samples_z.mean().norm().item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self._log_scalar("valid/latent_std_norm", samples_z.std().norm().item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        # Generate samples
        self.interpolant.set_device(batch['residue_mask'].device)
        atom37_traj, _, _, _ = self.interpolant.sample(
            batch_size, num_res, self.structure_model, residue_mask=batch['residue_mask'],
            trans_1=batch['trans_1'], rotmats_1=batch['rotmats_1'], 
            scaffold_mask=batch['scaffold_mask'], motif_mask=batch['motif_mask'],
            chain_index=batch['chain_index'], residue_index=batch['residue_index'], 
            fixed_structure_mask=batch['fixed_structure_mask'], aatype=batch['aatype'], group_mask=batch['motif_mask'],
            z_all=samples_z, pred_contact_map=pred_contact_map_sampled
        )
        samples = atom37_traj[-1].numpy()
        assert samples.shape == (batch_size, num_res, 37, 3)

        batch_metrics = []
        true_bb_pos = all_atom.atom37_from_trans_rot(batch['trans_1'], batch['rotmats_1'])
        for i in range(batch_size):
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

            # Visualize contact maps
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            # True contact map
            true_map = true_contacts[i].cpu().detach().numpy()
            ax1.imshow(true_map)
            ax1.set_title('True Contact Map')
            # Predicted contact map
            pred_map = pred_contact_map_sampled[i].cpu().detach().numpy()
            ax2.imshow(pred_map)  
            ax2.set_title('Predicted Contact Map')
            # Contact map difference
            diff_map = np.abs(true_map - pred_map)
            ax3.imshow(diff_map)
            ax3.set_title('Contact Map Difference')
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, 'contact_maps.png'))
            plt.close()

            # Cacluate fodling mettics
            ca_positions = torch.from_numpy(final_pos[:, ca_idx]).unsqueeze(0).to(batch['residue_mask'].device)
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

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=list(self.latent_diffusion.parameters()), 
            **self._exp_cfg.latent_diffusion_optimizer
        )
