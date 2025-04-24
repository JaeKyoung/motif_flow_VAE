from collections import defaultdict
import torch
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
import copy
from torch import autograd

from motifflow.data import all_atom
from motifflow.data import so3_utils
from motifflow.data import utils as du
# from motifflow.data.spatial_motif_prior import apply_spatial_motif_prior


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None
    
    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        if self._cfg.t_sampling == 'uniform':
            t = torch.rand(num_batch, device=self._device)
            return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t
        elif self._cfg.t_sampling == 'beta':
            beta_dist = torch.distributions.Beta(
                torch.tensor(1.9, device=self._device),
                torch.tensor(1.0, device=self._device)
            )
            t = beta_dist.sample((num_batch,))
            return t
        elif self._cfg.t_sampling == 'mixture':
            rand = torch.rand(num_batch, device=self._device)
            t = torch.empty(num_batch, device=self._device)
            mask_uniform = rand < 0.02
            if mask_uniform.any():
                t[mask_uniform] = torch.rand(mask_uniform.sum(), device=self._device) * (1 - 2 * self._cfg.min_t) + self._cfg.min_t
            mask_beta = ~mask_uniform
            if mask_beta.any():
                beta_dist = torch.distributions.Beta(
                    torch.tensor(1.9, device=self._device),
                    torch.tensor(1.0, device=self._device)
                )
                t[mask_beta] = beta_dist.sample((mask_beta.sum(),))
            return t
        elif self._cfg.t_sampling == 'reverse_mixture':
            rand = torch.rand(num_batch, device=self._device)
            t = torch.empty(num_batch, device=self._device)
            mask_uniform = rand < 0.02
            if mask_uniform.any():
                t[mask_uniform] = torch.rand(mask_uniform.sum(), device=self._device) * (1 - 2 * self._cfg.min_t) + self._cfg.min_t
            mask_beta = ~mask_uniform
            if mask_beta.any():
                beta_dist = torch.distributions.Beta(
                    torch.tensor(1.0, device=self._device),
                    torch.tensor(1.9, device=self._device)
                )
                t[mask_beta] = beta_dist.sample((mask_beta.sum(),))
            return t
        elif self._cfg.t_sampling == 'logit_normal':
            normal_dist = torch.distributions.Normal(
                torch.tensor(0.0, device=self._device),
                torch.tensor(1.0, device=self._device)
            )
            t = normal_dist.sample((num_batch,))
            t = 1 / (1 + torch.exp(-t))
            return t
        else:
            raise ValueError(f'Unknown t_sampling: {self._cfg.t_sampling}')
    
    def _corrupt_trans(self, trans_1, t, residue_mask, diffuse_mask):
        trans_nm_0 = _centered_gaussian(*residue_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        
        if self._trans_cfg.batch_ot:
            trans_0 = self._batch_ot(trans_0, trans_1, diffuse_mask)
        
        if self._trans_cfg.train_schedule == 'linear':
            trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        else:
            raise ValueError(
                f'Unknown trans schedule {self._trans_cfg.train_schedule}')
        
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * residue_mask[..., None]
    
    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        ) 
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
        
        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_rotmats(self, rotmats_1, t, residue_mask, diffuse_mask):
        num_batch, num_res = residue_mask.shape
        noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_batch*num_res).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)

        so3_schedule = self._rots_cfg.train_schedule
        if so3_schedule == 'exp':
            so3_t = 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif so3_schedule == 'linear':
            so3_t = t
        else:
            raise ValueError(f'Invalid schedule: {so3_schedule}')
            
        rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (rotmats_t * residue_mask[..., None, None] + identity[None, None] * (1 - residue_mask[..., None, None]))
        rotmats_t = _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)
        return rotmats_t

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1'] # Angstrom
        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']
        # [B, N]
        residue_mask = batch['residue_mask']
        scaffold_mask = batch['scaffold_mask']
        num_batch, _ = scaffold_mask.shape

        # [B,1]
        t = self.sample_t(num_batch)[:, None]
        so3_t = t
        r3_t = t
        noisy_batch['so3_t'] = so3_t
        noisy_batch['r3_t'] = r3_t

        diffuse_mask = torch.ones_like(residue_mask) # all 1 (all diffuse mask)
        
        # Apply corruptions
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(trans_1, r3_t, residue_mask, diffuse_mask)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t

        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(rotmats_1, so3_t, residue_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')
        noisy_batch['rotmats_t'] = rotmats_t

        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}'
            )
    
    def _trans_vector_field(self, t, trans_1, trans_t):
        if self._trans_cfg.sample_schedule == 'linear':
            trans_vf = (trans_1 - trans_t) / (1 - t)
            # trans_vf = (trans_1 - trans_t) / (1 - t + 1e-5)
        elif self._trans_cfg.sample_schedule == 'vpsde':
            bmin = self._trans_cfg.vpsde_bmin
            bmax = self._trans_cfg.vpsde_bmax
            bt = bmin + (bmax - bmin) * (1 - t) # scalar
            alpha_t = torch.exp(- bmin * (1 - t) - 0.5 * (1 - t)**2 * (bmax - bmin)) # scalar
            trans_vf = 0.5 * bt * trans_t \
                + 0.5 * bt * (torch.sqrt(alpha_t) * trans_1 - trans_t) / (1 - alpha_t).clamp(min=1e-5)
        else:
            raise ValueError(
                f'Invalid sample schedule: {self._trans_cfg.sample_schedule}'
            )
        return trans_vf

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t > 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
            # scaling = 1 / (1 - t + 1e-5)  
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        # TODO: Add in SDE.
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)


    def sample(
            self,
            num_batch,
            num_res,
            model,
            residue_mask=None,
            num_timesteps=None,
            trans_0=None,
            rotmats_0=None,
            trans_1=None,
            rotmats_1=None,
            scaffold_mask=None,
            motif_mask=None,
            chain_index=None,
            residue_index=None,
            fixed_structure_mask=None,
            aatype=None,
            group_mask=None,
            z_all=None,
            pred_contact_map=None,    
            verbose=False,
        ):
        
        residue_mask = residue_mask if residue_mask is not None else torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        trans_0 = trans_0 if trans_0 is not None else _centered_gaussian(num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        rotmats_0 = rotmats_0 if rotmats_0 is not None else _uniform_so3(num_batch, num_res, self._device)
        # Start from 1 res idx
        residue_index = residue_index if residue_index is not None else (torch.arange(num_res, device=self._device, dtype=torch.float32) + 1.0)[None].repeat(num_batch, 1)

        batch = {
            'residue_mask': residue_mask, # [B, N]
            'scaffold_mask': scaffold_mask, # [B, N]
            'motif_mask': motif_mask, # [B, N]
            'residue_index': residue_index, # [B, N]
            'chain_index': chain_index, # [B, N]
            'group_mask': group_mask, # [B, N]
            'fixed_structure_mask': fixed_structure_mask, # [B, N, N]
            'aatype': aatype, # [B, N]
            'trans_1': trans_1, # [B, N, 3]
            'rotmats_1': rotmats_1, # [B, N, 3, 3]
        }
        '''
        # Spatial motif-aligned prior and update batch
        if self._sample_cfg.use_spatial_motif_aligned_prior:
            batch = apply_spatial_motif_prior(batch, trans_0, rotmats_0)
        '''
        
        # Save Initial Batch
        initial_batch = batch

        '''
        # Set motif_mask
        if scaffold_mask is not None and trans_1 is not None and rotmats_1 is not None:
            motif_mask = ~batch['scaffold_mask'].bool().squeeze(0)
        else: # unconditional
            motif_mask = None

        # rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
        # trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)

        # trans_motif = trans_1[:,motif_mask]         # [1, motif_res, 3]
        # rotmats_motif = rotmats_1[:, motif_mask]    # [1, motif_res, 3, 3]
        '''

        # Set-up time steps
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []


        batch['z_all'] = z_all
        batch['pred_contact_map'] = pred_contact_map
        
        for i in range(len(ts) - 1):
            t_1, t_2 = ts[i], ts[i+1]
            if verbose: # and i % 1 == 0:
                print(f'{i=}, t={t_1.item():.2f}')
                print(torch.cuda.mem_get_info(trans_0.device), torch.cuda.memory_allocated(trans_0.device))

            # Run model
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            if self._trans_cfg.corrupt:
                batch['trans_t'] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError('Must provide trans_1 if not corrupting.')
                batch['trans_t'] = trans_1
                
            if self._rots_cfg.corrupt:
                batch['rotmats_t'] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError('Must provide rotmats_1 if not corrupting.')
                batch['rotmats_t'] = rotmats_1

            batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['so3_t'] = batch['t']
            batch['r3_t'] = batch['t']
            d_t = t_2 - t_1

            with torch.no_grad():
                model_out = model(batch)

            # Process model output
            pred_trans_1, pred_rotmats_1 = model_out['pred_trans'], model_out['pred_rotmats']
            clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
            
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1
                batch['rotmats_sc'] = pred_rotmats_1

            # Take reverse step
            trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
        
        # Final step (We only integrated to min_t)
        t_last = ts[-1]
        trans_t_last, rotmats_t_last = prot_traj[-1]

        if self._trans_cfg.corrupt:
            batch['trans_t'] = trans_t_last
        else:
            if trans_1 is None:
                raise ValueError('Must provide trans_1 if not corrupting.')
            batch['trans_t'] = trans_1
        if self._rots_cfg.corrupt:
            batch['rotmats_t'] = rotmats_t_last
        else:
            if rotmats_1 is None:
                raise ValueError('Must provide rotmats_1 if not corrupting.')
            batch['rotmats_t'] = rotmats_1

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_last
        
        with torch.no_grad():
            model_out = model(batch)
        
        pred_trans_last, pred_rotmats_last = model_out['pred_trans'], model_out['pred_rotmats']
        clean_traj.append((pred_trans_last.detach().cpu(), pred_rotmats_last.detach().cpu()))
        prot_traj.append((pred_trans_last, pred_rotmats_last))

        # Convert trajectories to atom37
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, residue_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, residue_mask)
        
        return atom37_traj, clean_atom37_traj, clean_traj, initial_batch