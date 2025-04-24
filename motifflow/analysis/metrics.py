""" Metrics. """
import mdtraj as md
import numpy as np
from openfold.np import residue_constants
from tmtools import tm_align
from motifflow.data import utils as du
import torch
from typing import Union, Optional

def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    #from https://github.com/aim-uofa/FADiff/blob/main/analysis/metrics.py
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2 

def calc_mdtraj_metrics(pdb_path):
    try:
        traj = md.load(pdb_path)
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_coil_percent = np.mean(pdb_ss == 'C')
        pdb_helix_percent = np.mean(pdb_ss == 'H')
        pdb_strand_percent = np.mean(pdb_ss == 'E')
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent 
        pdb_rg = md.compute_rg(traj)[0]
    except IndexError as e:
        print('Error in calc_mdtraj_metrics: {}'.format(e))
        pdb_ss_percent = 0.0
        pdb_coil_percent = 0.0
        pdb_helix_percent = 0.0
        pdb_strand_percent = 0.0
        pdb_rg = 0.0
    return {
        'non_coil_percent': pdb_ss_percent,
        'coil_percent': pdb_coil_percent,
        'helix_percent': pdb_helix_percent,
        'strand_percent': pdb_strand_percent,
        'radius_of_gyration': pdb_rg,
    }

def calc_ca_ca_metrics(ca_pos, bond_tol=0.1, clash_tol=1.0):
    ca_bond_dists = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca))
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + bond_tol))
    
    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
    clashes = inter_dists < clash_tol
    return {
        'ca_ca_deviation': ca_ca_dev,
        'ca_ca_valid_percent': ca_ca_valid,
        'num_ca_ca_clashes': np.sum(clashes),
    }

def calc_aligned_rmsd(pos_1, pos_2, use_torch=False):
    if use_torch:
        aligned_pos_1 = du.rigid_transform_3D(pos_1, pos_2, use_torch=True)[0]
        return torch.mean(torch.norm(aligned_pos_1 - pos_2, dim=-1))
    else:
        aligned_pos_1 = du.rigid_transform_3D(pos_1, pos_2, use_torch=False)[0]
        return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))

def cal_atom_lddt(
    pred_atom_pos: Union[np.ndarray, torch.Tensor],
    gt_atom_pos: Union[np.ndarray, torch.Tensor],
    atom_mask: Union[np.ndarray, torch.Tensor],
) -> float:
    """Calculate lDDT score of two sets of atom positions.

    Parameters
    ----------
    pred_positions: ndarray or FloatTensor, [N, L, 3]
        Predicted atom positions.
    gt_positions: ndarray or FloatTensor, [N, L, 3]
        Ground truth atom positions.
    atom_mask: ndarray or BoolTensor, [N, L]
        Mask of valid atoms.

    Returns
    -------
    lddt: float
        lDDT score.
    """
    assert pred_atom_pos.shape == gt_atom_pos.shape
    assert len(pred_atom_pos.shape) == 3
    assert pred_atom_pos.shape[-1] == 3
    assert atom_mask.shape == pred_atom_pos.shape[:-1]

    if isinstance(pred_atom_pos, np.ndarray):
        pred_atom_pos = torch.from_numpy(pred_atom_pos)
    if isinstance(gt_atom_pos, np.ndarray):
        gt_atom_pos = torch.from_numpy(gt_atom_pos)
    if isinstance(atom_mask, np.ndarray):
        atom_mask = torch.from_numpy(atom_mask)

    # distance matrix
    pred_dist = pred_atom_pos[None, :, None] - pred_atom_pos[:, None, :, None]
    pred_dist = torch.norm(pred_dist, dim=-1)
    gt_dist = gt_atom_pos[None, :, None] - gt_atom_pos[:, None, :, None]
    gt_dist = torch.norm(gt_dist, dim=-1)

    # get valid pairs
    pair_mask = atom_mask[:, None, :, None] * atom_mask[None, :, None, :]
    pair_mask &= gt_dist > 0
    pair_mask &= gt_dist < 15

    delta = torch.abs(pred_dist - gt_dist)
    lddt = torch.zeros_like(atom_mask).float()
    for distance_bin in [0.5, 1.0, 2.0, 4.0]:
        condition = ((delta <= distance_bin) * pair_mask).sum((1, 3)).float()
        condition /= pair_mask.sum((1, 3)) + 1e-8
        lddt += 0.25 * condition
    lddt = (lddt * atom_mask).sum() / (atom_mask.sum() + 1e-8)
    return float(lddt)

def cal_aligned_rmsd_2(
    prb_pos: Union[np.ndarray, torch.Tensor],  # (L, 3)
    ref_pos: Union[np.ndarray, torch.Tensor],  # (L, 3)
    res_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,  # (L)
) -> float:
    """Calculate RMSD of two sets of atom positions.
    Positions will be aligned before calculating RMSD.

    Parameters
    ----------
    prb_pos: ndarray or FloatTensor, (L, 3)
        Predicted atom positions.
    ref_pos: ndarray or FloatTensor, (L, 3)
        Reference atom positions.
    res_mask: ndarray or BoolTensor, (L)
        Mask of valid residues.

    Returns
    -------
    rmsd: float
        RMSD of two sets of atom positions.
    """
    if isinstance(prb_pos, torch.Tensor):
        prb_pos = du.to_numpy(prb_pos)
    if isinstance(ref_pos, torch.Tensor):
        ref_pos = du.to_numpy(ref_pos)
    if isinstance(res_mask, torch.Tensor):
        res_mask = du.to_numpy(res_mask)

    if res_mask is not None:
        non_gap_idx = np.where(~np.isnan(ref_pos).any(-1) & res_mask)[0]
    else:
        non_gap_idx = np.where(~np.isnan(ref_pos).any(-1))[0]
    if np.isnan(prb_pos[non_gap_idx]).any():
        raise ValueError(f"NaN in predicted positions. {prb_pos[non_gap_idx]}")
    aligned_prb_pos, _, _ = du.align_pos(prb_pos[non_gap_idx], ref_pos[non_gap_idx])
    rmsd = np.mean(np.linalg.norm(aligned_prb_pos - ref_pos[non_gap_idx], axis=-1))
    return rmsd.item()