"""Utility functions for experiments."""
import logging
import GPUtil
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import pandas as pd
import numpy as np
import os
import shutil
import glob
from biotite.sequence.io import fasta
from motifflow.analysis import utils as au
import subprocess


def dataset_creation(dataset_class, cfg, task):
    train_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=True,
    ) 
    eval_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=False,
    ) 
    return train_dataset, eval_dataset


def get_available_device(num_device):
    return GPUtil.getAvailable(order='memory', limit = 8)[:num_device]

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened

def save_traj(
        sample: np.ndarray,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        scaffold_mask: np.ndarray,
        chain_index: np.ndarray,
        output_dir: str,
        aatype = None,
        write_trajectories = True,
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
        aatype: [T, N, 21] amino acid probability vector trajectory.
        residue_mask: [N] residue mask.
        scaffold_mask: [N] which residues are scaffold.
        chain_index: [N] chain index.
        output_dir: where to save samples.
        aatype: [N, 21] amino acid type
        write_trajectories: whether to write trajectories.

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues and 0 for motif
        residues if there are any.
    """

    # Write sample.
    scaffold_mask = scaffold_mask.astype(bool)
    sample_path = os.path.join(output_dir, 'sample.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj.pdb')
    x0_traj_path = os.path.join(output_dir, 'x0_traj.pdb')

    # Use b-factors to specify which residues are scaffold.
    b_factors = np.tile((scaffold_mask * 100)[:, None], (1, 37))

    sample_path = au.write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        chain_index=chain_index,
        no_indexing=True,
        aatype=aatype,
    )
    if write_trajectories:
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors,
            chain_index=chain_index,
            no_indexing=True,
            aatype=aatype,
        )
        x0_traj_path = au.write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors,
            chain_index=chain_index,
            no_indexing=True,
            aatype=aatype
        )
    return {
        'sample_path': sample_path,
        'traj_path': prot_traj_path,
        'x0_traj_path': x0_traj_path,
    }

def calculate_diversity(output_dir, metrics_df, top_sample_csv, designable_csv_path, calculate_novlety=False):
    designable_samples = top_sample_csv[top_sample_csv.designable]
    designable_dir = os.path.join(output_dir, 'designable')
    os.makedirs(designable_dir, exist_ok=True)
    designable_txt = os.path.join(designable_dir, 'designable.txt')
    if os.path.exists(designable_txt):
        os.remove(designable_txt)
    with open(designable_txt, 'w') as f:
        for _, row in designable_samples.iterrows():
            sample_path = row.folded_path
            sample_name = f'sample_id_{row.sample_id}_length_{row.length}_mpnn_{row["Unnamed: 0"]}.pdb'
            write_path = os.path.join(designable_dir, sample_name)
            shutil.copy(sample_path, write_path)
            f.write(write_path+'\n')
    if metrics_df['Total designable'].iloc[0] <= 1:
        metrics_df['Clusters'] = metrics_df['Total designable'].iloc[0]
    else:
        add_diversity_metrics(designable_dir, metrics_df, designable_csv_path)
        if calculate_novlety:
            add_novelty_metrics(designable_dir, metrics_df, designable_csv_path)

def add_diversity_metrics(designable_dir, designable_csv, designable_csv_path):
    designable_txt = os.path.join(designable_dir, 'designable.txt')
    clusters = run_easy_cluster(designable_dir, designable_dir)
    designable_csv['Clusters'] = clusters
    designable_csv.to_csv(designable_csv_path, index=False)

def run_easy_cluster(designable_dir, output_dir):
    # designable_dir should be a directory with individual PDB files in it that we want to cluster
    # output_dir is where we are going to save the easy cluster output files

    # Returns the number of clusters
    try:
        easy_cluster_args = [
            'foldseek',
            'easy-cluster',
            designable_dir,
            os.path.join(output_dir, 'res'),
            os.path.join(output_dir, 'cluster_tmp'),
            '--alignment-type',
            '1',
            '--cov-mode',
            '0',
            '--min-seq-id',
            '0',
            '--tmscore-threshold',
            '0.5',
        ]
        process = subprocess.Popen(
            easy_cluster_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, _ = process.communicate()
        del stdout # We don't actually need the stdout, we will read the number of clusters from the output files
        shutil.rmtree(os.path.join(output_dir, 'cluster_tmp'))
        rep_seq_fasta = fasta.FastaFile.read(os.path.join(output_dir, 'res_rep_seq.fasta'))
        return len(rep_seq_fasta)
    
    except Exception as e:
        logging.warning(f"Error during clustering: {str(e)}. Returning 1 cluster")
        return 1

def add_novelty_metrics(designable_dir, designable_csv, designable_csv_path):
    run_easy_search(designable_dir, designable_dir)
    # designable_csv['Clusters'] = ?
    # designable_csv.to_csv(designable_csv_path, index=False)

def run_easy_search(designable_dir, output_dir):
    # foldseek easy-search <path_sample> <database_path> <out_file> <tmp_path> --alignment-type 1 --exhaustive-search --tmscore-threshold 0.0 --max-seqs 10000000000 --format-output query,target,alntmscore,lddt
    try:
        easy_search_args = [
            'foldseek',
            'easy-search',
            designable_dir,
            '/home/fullmoon/projects/ZFN/foldseek/pdb/pdb', # /home/fullmoon/projects/ZFN/foldseek/afdb/afdb
            os.path.join(output_dir, 'easy_search_output.tsv'),
            os.path.join(output_dir, 'tmpFolder'),
            '--alignment-type',
            '1',
            '--exhaustive-search',
            '--tmscore-threshold',
            '0.0',
            '--max-seqs',
            '10000000000',
            '--format-output',
            'query,target,alntmscore,lddt'
        ]
        process = subprocess.Popen(
            easy_search_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, _ = process.communicate()
        del stdout # We don't actually need the stdout, we will read the alntmscore from the output files
        shutil.rmtree(os.path.join(output_dir, 'tmpFolder'))
        easy_search_output = pd.read_csv(os.path.join(output_dir, 'easy_search_output.tsv'), sep="\t")
        easy_search_output.columns = ["query", "target", "alntmscore", "lddt"]
        max_tm_scores = easy_search_output.loc[easy_search_output.groupby("query")["alntmscore"].idxmax()][["query", "alntmscore", "target", "lddt"]].reset_index(drop=True)
        max_tm_scores.to_csv(os.path.join(output_dir, 'max_tm_scores.tsv'), sep="\t", index=False)
    
    except Exception as e:
        logging.warning(f"Error during searching: {str(e)}.")

def get_all_top_samples(output_dir, csv_fname='*/top_sample.csv'):
    all_csv_paths = glob.glob(os.path.join(output_dir, csv_fname), recursive=True)
    top_sample_csv = pd.concat([pd.read_csv(x) for x in all_csv_paths])
    top_sample_csv.to_csv(
        os.path.join(output_dir, 'all_top_samples.csv'), index=False)
    return top_sample_csv