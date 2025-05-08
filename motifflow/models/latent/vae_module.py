from typing import Dict
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from motifflow.models.base_module import BaseModule
from motifflow.models.latent.utils import (
    compute_contact_map,
    compute_vae_loss,
    calculate_kl_beta,
    compute_vae_auc
)
from motifflow.data import utils as du

class ProteinVAEModule(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        # KL Annealing
        self.beta_max = self._exp_cfg.training.beta_max
        self.warmup_epochs = self._exp_cfg.training.warmup_epochs
        self.total_annealing_epochs = self._exp_cfg.training.total_annealing_epochs
        self.current_beta = 0.0

    def on_train_epoch_start(self):
        self.current_beta = calculate_kl_beta(
            current_epoch=self.current_epoch, 
            total_annealing_epochs=self.total_annealing_epochs,
            beta_max=self.beta_max,
            warmup_epochs=self.warmup_epochs
        )
        self.log('train/kl_beta', self.current_beta, on_step=False, on_epoch=True, prog_bar=False)


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        z_all, mean_all, logvar_all = self.encoder(
            batch['trans_1'], batch['rotmats_1'], batch['aatype'],
            batch['residue_mask'], batch['residue_mask'],
            batch["residue_index"], batch["chain_index"]
        )
        pred_contact_map, pred_contact_logits = self.decoder(z_all, batch['residue_mask'], batch['residue_mask'])
        
        # Compute losses
        true_contacts = compute_contact_map(batch['trans_1'], batch['residue_mask'])
        recon_loss, kl_loss = compute_vae_loss(pred_contact_logits, true_contacts, mean_all, logvar_all, batch['residue_mask'])
        roc_auc, pr_auc = compute_vae_auc(pred_contact_map, true_contacts, batch['residue_mask'])
        total_loss = (recon_loss + self.current_beta * kl_loss)

        # Log metrics
        metrics = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mean' : np.mean(du.to_numpy(mean_all)),
            'logvar' : np.mean(du.to_numpy(logvar_all)),
            'roc_auc': np.mean(roc_auc),
            'pr_auc': np.mean(pr_auc)
        }
        batch_size = batch['trans_1'].shape[0]
        for name, value in metrics.items():
            self._log_scalar(f'train/{name}', value, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch_size)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=list(self.encoder.parameters()) + list(self.decoder.parameters()),
            **self._exp_cfg.optimizer
        )
        return optimizer

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # Forward pass
        z_all, mean_all, logvar_all = self.encoder(
            batch['trans_1'], batch['rotmats_1'], batch['aatype'],
            batch['residue_mask'], batch['residue_mask'],
            batch["residue_index"], batch["chain_index"]
        )
        pred_contact_map, pred_contact_logits = self.decoder(z_all, batch['residue_mask'], batch['residue_mask'])
        
        # Compute losses
        true_contacts = compute_contact_map(batch['trans_1'], batch['residue_mask'])
        recon_loss, kl_loss = compute_vae_loss(
            pred_contact_logits, true_contacts, mean_all, logvar_all, batch['residue_mask']
        )
        roc_auc, pr_auc = compute_vae_auc(pred_contact_map, true_contacts, batch['residue_mask'])
        total_loss = (recon_loss + self.current_beta * kl_loss)
    
        metrics = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mean' : np.mean(du.to_numpy(mean_all)),
            'logvar' : np.mean(du.to_numpy(logvar_all)),
            'roc_auc': np.mean(roc_auc),
            'pr_auc': np.mean(pr_auc)
        }

        batch_size = batch['trans_1'].shape[0]
        num_res = batch['trans_1'].shape[1]

        # Visualize contact maps
        for i in range(batch_size):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f'sample_{batch["csv_idx"][i].item()}_{batch["pdb_name"][i]}_idx_{batch_idx}_len_{num_res}'
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Plot contact maps
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            true_map = true_contacts[i].cpu().detach().numpy()
            pred_map = pred_contact_map[i].cpu().detach().numpy()
            diff_map = np.abs(true_map - pred_map)

            ax1.imshow(true_map)
            ax1.set_title('True Contact Map')
            
            ax2.imshow(pred_map)  
            ax2.set_title('Predicted Contact Map')
            
            ax3.imshow(diff_map)
            ax3.set_title('Contact Map Difference')
            
            plt.savefig(os.path.join(sample_dir, 'contact_maps.png'))
            plt.close()

        batch_metrics = pd.DataFrame([metrics])
        self.validation_epoch_metrics.append(batch_metrics)