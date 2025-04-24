import numpy as np
import os
import re
import shutil

import pandas as pd
import torch
from biotite.sequence.io import fasta

from motifflow.analysis import metrics
from motifflow.data import utils as du
from motifflow.experiments import utils as eu
from motifflow.models.structure import utils as msu
from openfold.utils.superimposition import superimpose


def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses

def process_folded_outputs(sample_path, folded_output, true_bb_pos, motif_mask, group_mask, chain_index):
    mpnn_results = {
        'header': [],
        'sequence': [],
        'mean_plddt': [],
        'folded_path': [],
        'tm_score_fold_sample': [],
        'bb_rmsd_fold_sample': [],
        'rmsd_2_fold_sample': [],
        'mean_pae': [],
    }
    # Chain A (chain_index==1) part
    chain_a_mask = (chain_index.squeeze() == 1).cpu().numpy()
    
    # Filtering only chain A part
    if true_bb_pos is not None:
        expanded_mask = np.repeat(chain_a_mask, 3) # for N, CA, C
        true_bb_pos = true_bb_pos[expanded_mask]
    if motif_mask is not None:
        motif_mask = motif_mask.squeeze()[chain_a_mask].unsqueeze(0)
    if group_mask is not None:
        group_mask = group_mask.squeeze()[chain_a_mask].unsqueeze(0)

    if true_bb_pos is not None: # motif scaffolding
        mpnn_results['motif_bb_rmsd_sample_gt'] = []
        mpnn_results['motif_bb_rmsd_fold_gt'] = []
        mpnn_results['motif_rmsd_2_fold_sample'] = []
        mpnn_results['motif_bb_rmsd_fold_sample'] = []

    sample_feats = du.parse_pdb_feats('sample', sample_path)
    sample_ca_pos = sample_feats['bb_positions'] # already only chain A
    sample_bb_pos = sample_feats['atom_positions'][:, :3].reshape(-1, 3)

    def _calc_ca_rmsd(mask, folded_ca_pos):
        return superimpose(
            torch.tensor(sample_ca_pos)[None],
            torch.tensor(folded_ca_pos[None]),
            mask
        )[1].rmsd[0].item()
    
    def _calc_bb_rmsd(mask, ref_bb_pos, target_bb_pos):
        _, aligned_rmsd = superimpose(
            torch.tensor(ref_bb_pos, device=mask.device)[None],
            torch.tensor(target_bb_pos, device=mask.device)[None],
            mask[:, None].repeat(1, 3).reshape(-1)
        ) #reference, coords, mask
        return aligned_rmsd.item()

    for _, row in folded_output.iterrows():
        folded_feats = du.parse_pdb_feats('folded', row.folded_path)
        seq = du.aatype_to_seq(folded_feats['aatype'])
        folded_ca_pos = folded_feats['bb_positions']
        folded_bb_pos = folded_feats['atom_positions'][:, :3].reshape(-1, 3)
        res_mask = torch.ones(folded_ca_pos.shape[0])
        motif_mask = motif_mask.squeeze(0)

        # tm_score 계산
        _, tm_score = metrics.calc_tm_score(sample_ca_pos, folded_ca_pos, seq, seq)
        mpnn_results['tm_score_fold_sample'].append(tm_score)

        # all RMSD
        bb_rmsd = _calc_bb_rmsd(res_mask, sample_bb_pos, folded_bb_pos)
        mpnn_results['bb_rmsd_fold_sample'].append(bb_rmsd)
        rmsd_2 = metrics.calc_aligned_rmsd(sample_ca_pos, folded_ca_pos)
        mpnn_results['rmsd_2_fold_sample'].append(rmsd_2)

        if true_bb_pos is not None: # motif scaffolding
            # motif RMSD
            group_ids = torch.unique(group_mask[group_mask > 0])  # 0 제외한 유니크한 그룹 ID
            # 각 그룹별 RMSD 저장
            motif_bb_rmsd_fold_sample = []
            motif_rmsd_2 = []
            motif_bb_rmsd_sample_gt = []
            motif_bb_rmsd_fold_gt = []
            
            for group_id in group_ids:
                curr_group_mask = (group_mask == group_id).squeeze(0)
                
                curr_group_rmsd_fold_sample = _calc_bb_rmsd(curr_group_mask,sample_bb_pos,folded_bb_pos)
                motif_bb_rmsd_fold_sample.append(curr_group_rmsd_fold_sample)  

                curr_group_motif_bb_rmsd_sample_gt = _calc_bb_rmsd(curr_group_mask, true_bb_pos, sample_bb_pos)
                motif_bb_rmsd_sample_gt.append(curr_group_motif_bb_rmsd_sample_gt)

                curr_group_motif_fold_model_bb_rmsd_fold_gt = _calc_bb_rmsd(curr_group_mask, true_bb_pos, folded_bb_pos)
                motif_bb_rmsd_fold_gt.append(curr_group_motif_fold_model_bb_rmsd_fold_gt)

                curr_group_mask_np = du.to_numpy(curr_group_mask).astype(bool)
                curr_group_rmsd_2 = metrics.calc_aligned_rmsd(sample_ca_pos[curr_group_mask_np], folded_ca_pos[curr_group_mask_np])
                motif_rmsd_2.append(curr_group_rmsd_2)

            # 그룹별 평균 RMSD 계산
            avg_motif_bb_rmsd_fold_sample = sum(motif_bb_rmsd_fold_sample) / len(motif_bb_rmsd_fold_sample)
            mpnn_results['motif_bb_rmsd_fold_sample'].append(avg_motif_bb_rmsd_fold_sample)
            avg_motif_bb_rmsd_sample_gt = sum(motif_bb_rmsd_sample_gt) / len(motif_bb_rmsd_sample_gt)
            mpnn_results['motif_bb_rmsd_sample_gt'].append(avg_motif_bb_rmsd_sample_gt)
            avg_motif_bb_rmsd_fold_gt = sum(motif_bb_rmsd_fold_gt) / len(motif_bb_rmsd_fold_gt)
            mpnn_results['motif_bb_rmsd_fold_gt'].append(avg_motif_bb_rmsd_fold_gt)
            avg_motif_rmsd_2 = sum(motif_rmsd_2) / len(motif_rmsd_2)
            mpnn_results['motif_rmsd_2_fold_sample'].append(avg_motif_rmsd_2)

        mpnn_results['folded_path'].append(row.folded_path)
        mpnn_results['header'].append(row.header)
        mpnn_results['sequence'].append(seq)
        mpnn_results['mean_plddt'].append(row.plddt)
        mpnn_results['mean_pae'].append(row.pae)
    mpnn_results = pd.DataFrame(mpnn_results)
    mpnn_results['sample_path'] = sample_path
    return mpnn_results

def extract_clusters_from_maxcluster_out(file_path):
    # Extracts cluster information from the stdout of a maxcluster run
    cluster_to_paths = {}
    paths_to_cluster = {}
    read_mode = False
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line == "INFO  : Item     Cluster\n":
                read_mode = True
                continue

            if line == "INFO  : ======================================\n":
                read_mode = False

            if read_mode:
                # Define a regex pattern to match the second number and the path
                pattern = r"INFO\s+:\s+\d+\s:\s+(\d+)\s+(\S+)"

                # Use re.search to find the first match in the string
                match = re.search(pattern, line)

                # Check if a match is found
                if match:
                    # Extract the second number and the path
                    cluster_id = match.group(1)
                    path = match.group(2)
                    if cluster_id not in cluster_to_paths:
                        cluster_to_paths[cluster_id] = [path]
                    else:
                        cluster_to_paths[cluster_id].append(path)
                    paths_to_cluster[path] = cluster_id

                else:
                    raise ValueError(f"Could not parse line: {line}")

    return cluster_to_paths, paths_to_cluster

def run_pmpnn_post_processing(folding_model, write_dir, pdb_input_path, motif_mask, chain_index):
    folding_model.run_proteinmpnn(write_dir, pdb_input_path, motif_mask, chain_index)
    mpnn_fasta_path = os.path.join(write_dir, 'seqs', os.path.basename(pdb_input_path).replace('.pdb', '.fa')) # A chain only sequence
    fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
    all_header_seqs = [
        (f'pmpnn_seq_{i}', seq) for i, (_, seq) in enumerate(fasta_seqs.items())
        if i > 0
    ]
    modified_fasta_path = mpnn_fasta_path.replace('.fa', '_modified.fasta')
    fasta.FastaFile.write_iter(modified_fasta_path, all_header_seqs)
    return modified_fasta_path

def compute_sample_metrics(model_traj, bb_traj,
                            true_bb_pos, aatype, motif_mask, group_mask, scaffold_mask, chain_index,
                            sample_id, sample_length, sample_dir, 
                            folding_model, self_consistency_metric, write_sample_trajectories):
    
    noisy_traj_length, sample_length, _, _ = bb_traj.shape
    clean_traj_length = model_traj.shape[0]
    assert bb_traj.shape == (noisy_traj_length, sample_length, 37, 3)
    assert model_traj.shape == (clean_traj_length, sample_length, 37, 3)

    os.makedirs(sample_dir, exist_ok=True)
    
    traj_paths = eu.save_traj(
            bb_traj[-1],
            bb_traj,
            np.flip(model_traj, axis=0),
            # np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
            du.to_numpy(scaffold_mask)[0],
            chain_index=chain_index[0],
            output_dir=sample_dir,
            aatype=aatype,
            write_trajectories = write_sample_trajectories
        )

    # Run PMPNN to get sequences
    sc_output_dir = os.path.join(sample_dir, 'self_consistency')
    os.makedirs(sc_output_dir, exist_ok=True)
    pdb_path = traj_paths['sample_path']
    pmpnn_pdb_path = os.path.join(sc_output_dir, os.path.basename(pdb_path))
    shutil.copy(pdb_path, pmpnn_pdb_path)
    pmpnn_fasta_path = run_pmpnn_post_processing(folding_model, sc_output_dir, pmpnn_pdb_path, motif_mask, chain_index) # only A chain sequence design and output

    # Run ESMfold on each ProteinMPNN sequence (only A chain) and calculate metrics.
    folded_dir = os.path.join(sc_output_dir, 'folded')
    if os.path.exists(folded_dir):
        shutil.rmtree(folded_dir)
    os.makedirs(folded_dir, exist_ok=False)
    folded_output = folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)
    # Post Processing of folded structure
    mpnn_esm_results = msu.process_folded_outputs(pdb_path, folded_output, true_bb_pos, motif_mask, group_mask, chain_index)

    # Save results to CSV
    mpnn_esm_results.to_csv(os.path.join(sample_dir, 'sc_results.csv'))
    mpnn_esm_results['length'] = sample_length
    mpnn_esm_results['sample_id'] = sample_id
    del mpnn_esm_results['header']
    del mpnn_esm_results['sequence']

    # Select the top sample
    # scbb_rmsd top
    if self_consistency_metric == 'scRMSD':
        top_sample = mpnn_esm_results.sort_values('bb_rmsd_fold_sample', ascending=True).iloc[:1]
    elif self_consistency_metric == 'scTM':
        top_sample = mpnn_esm_results.sort_values('tm_score_fold_sample', ascending=False).iloc[:1]
    else:
        raise ValueError(f'Unknown top self-consistency scoring {self_consistency_metric}')
    
    # Compute secondary structure metrics
    sample_dict = top_sample.iloc[0].to_dict()
    ss_metrics = metrics.calc_mdtraj_metrics(sample_dict['sample_path'])
    top_sample['helix_percent'] = ss_metrics['helix_percent']
    top_sample['strand_percent'] = ss_metrics['strand_percent']

    return top_sample