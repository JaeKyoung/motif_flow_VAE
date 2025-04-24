import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from motifflow.models.base_module import BaseModule
from copy import deepcopy
from motifflow.modules.diffusion.create_diffusion import create_diffusion
from motifflow.models.structure.flow_module import FlowModel
from motifflow.data.interpolant import Interpolant
from motifflow.models.latent.latent_diffusion import ProteinLatentDiT
from collections import OrderedDict

#########################################################
# ProteinLatentDiffusionModule
#########################################################

@torch.no_grad()
def update_ema(self, decay: float = 0.9999):
    """
    EMA update for the model.
    """
    ema_params = dict(self.ema.named_parameters())
    model_params = dict(self.latent_diffusion.named_parameters())

    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)

class ProteinLatentDiffusionModule(BaseModule):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Set up VAE
        if self._exp_cfg.joint_checkpoint:  # config에 vae checkpoint path 추가 필요
            # Load VAE weights from checkpoint
            vae_state_dict = torch.load(self._exp_cfg.joint_checkpoint)['state_dict']
            encoder_state_dict = {k.replace('encoder.', ''): v for k, v in vae_state_dict.items()
                                if k.startswith('encoder.')}
            decoder_state_dict = {k.replace('decoder.', ''): v for k, v in vae_state_dict.items()
                                if k.startswith('decoder.')}
            self.encoder.load_state_dict(encoder_state_dict)
            self.decoder.load_state_dict(decoder_state_dict)
            self._print_logger.info(f"Loaded encoder and decoder weights from {self._exp_cfg.joint_checkpoint}")
            
          
            if self._cfg.latent_diffusion.use_structure_loss:
                structure_model_state_dict = {k.replace('structure_model.', ''): v for k, v in vae_state_dict.items()   
                                    if k.startswith('structure_model.')}
                self.structure_model.load_state_dict(structure_model_state_dict)
                
        else:
            self._print_logger.info("No checkpoint provided")
        
        
        # Set up latent diffusion model
        self.latent_diffusion = ProteinLatentDiT(self._cfg.latent_diffusion)
        self.diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        # EMA
        self.ema = deepcopy(self.latent_diffusion)
        for p in self.ema.parameters():
            p.requires_grad = False
        
        # Set up structure model and interpolant
        if self._cfg.latent_diffusion.use_structure_loss:
            self.structure_model = FlowModel(self._cfg.structure_model)
            self.interpolant = Interpolant(self._cfg.interpolant)


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        '''
        Training latent diffusion model
        '''
        
        B, num_res, _ = batch['trans_1'].shape
        device = batch['trans_1'].device
        
        # Encode the input (motif) region
        with torch.no_grad():
            z_0, mean, logvar = self.encoder.encode(
                batch['trans_1'], batch['rotmats_1'], batch['aatype'],
                batch['motif_mask'], batch['residue_mask'],
                batch["residue_index"], batch["chain_index"]
            )
        
        # 쓰바 모르게따

        # Sample a random timestep
        t = torch.randint(0, self._cfg.latent_diffusion.num_timesteps, (B,), device=z_0.device)
        t = self.diffusion.num_timesteps - 1 - t
        noise = torch.randn_like(z_0)

        # Forward diffusion process
        z_t = self.diffusion.q_sample(z_0, t, noise=noise)  # Add noise to z_0

        # Predict noise
        eps_pred = self.latent_diffusion(z_t, t, batch['motif_mask']) # (B, N, D)

        # Compute loss
        loss_diffusion = F.mse_loss(eps_pred, noise)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1).mean()
        total_loss = loss_diffusion + self._training_cfg.kl_weight * kl_loss

        if self._cfg.latent_diffusion.use_structure_loss: # End-to-End training (slow)
            batch["z_all"] = z_0  # 원래 ground-truth z₀
            batch["pred_contact_map"] = self.decoder.decode(z_0, batch['residue_mask'], batch['motif_mask'])

            # Generate samples
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

        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/diffusion_loss", loss_diffusion, prog_bar=True)
        self.log("train/kl_loss", kl_loss, prog_bar=True)

        update_ema()

        return total_loss
    
    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return 0
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=list(self.latent_diffusion.parameters()), 
            **self._exp_cfg.latent_diffusion_optimizer
        )
