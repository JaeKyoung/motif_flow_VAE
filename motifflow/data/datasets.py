import abc
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random
import os

from torch.utils.data import Dataset
from motifflow.data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def _rog_filter(df, quantile):
    # Calculate radius of gyration quantiles for each sequence length
    y_quant = pd.pivot_table(
        df,
        values='radius_gyration', 
        index='modeled_seq_len',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit a polynomial regressor to the quantiles
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff values for all sequence lengths
    max_len = df.modeled_seq_len.max()
    pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
    # Add a small buffer
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

    # Filter the dataframe based on the calculated cutoffs
    row_rog_cutoffs = df.modeled_seq_len.map(lambda x: pred_y[x-1])
    return df[df.radius_gyration < row_rog_cutoffs]


def _length_filter(data_csv, min_res, max_res):
    # Filter the dataframe based on sequence length
    return data_csv[
        (data_csv.modeled_seq_len >= min_res) &
        (data_csv.modeled_seq_len <= max_res)
    ]

def _plddt_percent_filter(data_csv, min_plddt_percent):
    # Filter the dataframe based on the percentage of confident pLDDT scores
    return data_csv[data_csv.num_confident_plddt > min_plddt_percent]


def _max_coil_filter(data_csv, max_coil_percent):
    # Filter the dataframe based on the maximum allowed percentage of coil structure
    return data_csv[data_csv.coil_percent <= max_coil_percent]


def _process_csv_row(processed_file_path):
    # Load and process features from a pickle file
    # In pickle file,
    # atom_positions, aatype, atom_mask, residue_index, chain_index,
    # b_factors, bb_mask, bb_positions, modeled_idx
    processed_feats = du.read_pkl(processed_file_path)
    # Center the position
    processed_feats = du.parse_chain_feats(processed_feats)

    # Extract only the modeled residues
    modeled_idx = processed_feats['modeled_idx']
    min_idx, max_idx = np.min(modeled_idx), np.max(modeled_idx)
    del processed_feats['modeled_idx']
    processed_feats = tree.map_structure(lambda x: x[min_idx:(max_idx+1)], processed_feats)

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
    }
   
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    res_plddt = processed_feats['b_factors'][:, 1]
    res_mask = torch.tensor(processed_feats['bb_mask']).int()

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices among 1~5 but must include 1.
    chain_idx = processed_feats['chain_index']
    res_idx = processed_feats['residue_index']
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()

    n_chains = len(all_chain_idx)
    # 1을 포함하고 나머지는 2~6에서 랜덤하게 선택 (max 6개)
    if n_chains == 1:
        selected_numbers = [1]
    else:
        selected_numbers = [1] + random.sample(range(2, 7), n_chains - 1)
        # 전체 순서를 섞어서 1의 위치도 랜덤하게 만듦
        random.shuffle(selected_numbers)

    for i, chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)
        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
        new_res_idx += (res_idx - chain_min_idx + 1) * chain_mask
        new_chain_idx += selected_numbers[i] * chain_mask

    ''' TODO
    # Chain 1이 앞에 있도록 Permutation
    unique_chains = np.unique(new_chain_idx)
    chain_1_idx = np.where(unique_chains == 1)[0][0]
    
    # Chain 1이 첫번째로 오도록 순서 변경
    if chain_1_idx != 0:
        # Chain 1과 첫번째 체인의 위치를 교환
        chain_1_mask = (new_chain_idx == 1)
        first_chain_mask = (new_chain_idx == unique_chains[0])
        
        # Chain 1을 첫번째 체인 값으로 변경
        new_chain_idx[chain_1_mask] = unique_chains[0]
        # 첫번째 체인을 1로 변경  
        new_chain_idx[first_chain_mask] = 1
        
        # Residue index도 함께 교환
        chain_1_res_idx = new_res_idx[chain_1_mask].copy()
        first_chain_res_idx = new_res_idx[first_chain_mask].copy()
        
        new_res_idx[chain_1_mask] = first_chain_res_idx
        new_res_idx[first_chain_mask] = chain_1_res_idx
    '''
    
    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f'Found NaNs in {processed_file_path}')
    
    return {
        'all_atom_positions': chain_feats['all_atom_positions'],
        'residue_plddt': res_plddt,
        'aatype': chain_feats['aatype'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'residue_mask': res_mask,
        'chain_index': new_chain_idx,
        'residue_index': new_res_idx,
    }


def _add_plddt_mask(feats, plddt_threshold):
    # Add a mask based on pLDDT scores
    feats['plddt_mask'] = torch.tensor(feats['residue_plddt'] > plddt_threshold).int()


def _read_clusters(cluster_path):
    # Read cluster information from a file
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i, line in enumerate(f):
            for chain in line.split(' '):
                pdb = chain.split('_')[0]
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster


class BaseDataset(Dataset):
    def __init__(self, *, dataset_cfg, is_training, task,):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values('modeled_seq_len', ascending=False)
        self._create_split(metadata_csv)
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        # for debugging
        # metadata_csv.to_csv('metadata_filtered.csv', index=False)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg
    
    def __len__(self):
        return len(self.csv)

    @abc.abstractmethod
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        pass

    def _create_split(self, data_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = data_csv
            self._log.info(f'Training: {len(self.csv)} examples')
        else:
            if self._dataset_cfg.max_eval_length is None:
                eval_lengths = data_csv.modeled_seq_len
            else:
                eval_lengths = data_csv.modeled_seq_len[data_csv.modeled_seq_len <= self._dataset_cfg.max_eval_length]
            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(0.0, 1.0, self.dataset_cfg.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = data_csv[data_csv.modeled_seq_len.isin(eval_lengths)]
            
            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self.dataset_cfg.samples_per_eval_length,
                replace=True,
                random_state=123
            )
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv
            self._log.info(f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')
        self.csv['index'] = list(range(len(self.csv)))

    def process_csv_row(self, csv_row):
        # Process a single row from the CSV file
        path = csv_row['processed_path']
        seq_len = csv_row['modeled_seq_len']
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        processed_row = _process_csv_row(path)
        if use_cache:
            self._cache[path] = processed_row
        return processed_row
    
    def _sample_motif_mask(self, feats, rng): 
        #TODO : change Function to select motifs from residue mask
        # Generate a scaffold mask for inpainting
        residue_index = feats['residue_index']
        num_res = residue_index.shape[0]
        chain_idx = feats['chain_index']

        # Create mask for chain_idx ==1
        chain1_mask = (chain_idx == 1).astype(int)
        chain1_res = chain1_mask.sum()

        # Calculate motif sizes based on chain 1 length
        min_motif_size = max(1, int(self._dataset_cfg.min_motif_percent * chain1_res))
        max_motif_size = max(min_motif_size, int(self._dataset_cfg.max_motif_percent * chain1_res))
        # print(f"[Debug] min_motif_size: {min_motif_size}, chain1_res: {chain1_res}, min_percent: {self._dataset_cfg.min_motif_percent}")
        # print(f"[Debug] max_motif_size: {max_motif_size}, chain1_res: {chain1_res}, max_percent: {self._dataset_cfg.max_motif_percent}")
       
        # Sample the total number of residues that will be used as the motif.
        total_motif_size = self._rng.integers(low=min_motif_size, high=max_motif_size+1)
        # print(f"[Debug] total_motif_size: {total_motif_size}")

        # Sample motifs at different locations.
        num_motifs = self._rng.integers(
            low=1, 
            high=min(self._dataset_cfg.motif_max_n_seg, total_motif_size)+1
        )

        # sample motif mask
        indices = sorted(self._rng.choice(total_motif_size - 1, num_motifs - 1, replace=False) + 1 )
        indices = [0] + indices + [total_motif_size]
        motif_seg_lens = [indices[i+1] - indices[i] for i in range(num_motifs)]

        # Generate motif mask only for chain 1
        segs = [''.join(['1'] * l) for l in motif_seg_lens]
        segs.extend(['0'] * (chain1_res - total_motif_size))
        self._rng.shuffle(segs)
        chain1_motif_mask = torch.tensor([int(elt) for elt in ''.join(segs)], dtype=torch.float)
        
        # Apply chain1 motif mask only to residues where chain_index == 1
        motif_mask = torch.ones(num_res, dtype=torch.float)
        chain1_residues = torch.where(torch.from_numpy(chain_idx) == 1)[0]
        motif_mask[chain1_residues] = chain1_motif_mask
        motif_mask = motif_mask * feats['residue_mask']

        # Generate motif_structure_mask
        motif_structure_mask = motif_mask[:, np.newaxis] * motif_mask[np.newaxis, :]

        return motif_mask, motif_structure_mask
        

    def setup_inpainting(self, feats, rng):
        # Set up inpainting mask
        motif_mask, fixed_structure_mask = self._sample_motif_mask(feats, rng)
        scaffold_mask = ( 1 - motif_mask ) * feats['residue_mask']
        
        if 'plddt_mask' in feats:
            motif_mask = motif_mask * feats['plddt_mask']
            scaffold_mask = scaffold_mask * feats['plddt_mask']
        
        feats['motif_mask'] = motif_mask
        feats['scaffold_mask'] = scaffold_mask
        feats['fixed_structure_mask'] = fixed_structure_mask
        
        return feats

    def __getitem__(self, row_idx):
        # Process data example.
        # Get a single item from the dataset
        """
        feats:
            'trans_1': Translation vectors (motif-centered)
            'rotmats_1': Rotation matrices
            'residue_index': Residue indices
            'chain_index': Chain indices
            'residue_mask': Residue mask
            'ressidue_plddt': Per-residue pLDDT scores
            'plddt_mask': pLDDT mask (all 1 if no use)
            'aatype': Amino acid types as int

            For Motif Scaffolding
            'scaffold_mask': scaffolding mask
            'fixed_structure_mask'

            For debugging
            'csv_idx': CSV index
            'pdb_name': PDB name
        """
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)

        if self._dataset_cfg.add_plddt_mask:
            _add_plddt_mask(feats, self._dataset_cfg.min_plddt_threshold)
        else:
            feats['plddt_mask'] = torch.ones_like(feats['residue_mask'])

        if self.task == 'inpainting':
            rng = self._rng if self.is_training else np.random.default_rng(seed=123)
            feats = self.setup_inpainting(feats, rng)
            # Center based on motif locations 
            motif_mask = feats['motif_mask']
            motif_chain1_mask = torch.tensor(feats['chain_index'] == 1) * motif_mask
            trans_1 = feats['trans_1']
            motif_1_ch1 = trans_1 * motif_chain1_mask[:, None]
            motif_com = torch.sum(motif_1_ch1, dim=0) / (torch.sum(motif_chain1_mask) + 1)
            trans_1 -= motif_com[None, :]
            feats['trans_1'] = trans_1
        else:
            raise ValueError(f'Unknown task {self.task}')
        feats['scaffold_mask'] = feats['scaffold_mask'].int()
        feats['fixed_structure_mask'] = feats['fixed_structure_mask'].int()
        
        # Storing the csv index is helpful for debugging.
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx
        feats['pdb_name'] = csv_row['pdb_name']

        return feats


class SCOPeDataset(BaseDataset):
    def _filter_metadata(self, raw_csv):
        # Filter metadata for SCOPE dataset
        filter_cfg = self.dataset_cfg.filter
        data_csv = _length_filter(raw_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
        data_csv['oligomeric_detail'] = 'monomeric'
        return data_csv


class PDBDataset(BaseDataset):
    def __init__(self, *, dataset_cfg, is_training, task,):

        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

        # Process clusters
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values('modeled_seq_len', ascending=False)

        self._pdb_to_cluster = _read_clusters(self._dataset_cfg.cluster_path)
        self._max_cluster = max(self._pdb_to_cluster.values())
        self._missing_pdbs = 0
        
        def cluster_lookup(pdb):
            # Assign cluster IDs to PDB entries
            pdb = pdb.upper()
            if pdb not in self._pdb_to_cluster:
                self._pdb_to_cluster[pdb] = self._max_cluster + 1
                self._max_cluster += 1
                self._missing_pdbs += 1
            return self._pdb_to_cluster[pdb]
        
        metadata_csv['cluster'] = metadata_csv['pdb_name'].map(cluster_lookup)
        # metadata_csv.to_csv('PDB_metadata_filter_cluster.csv', index=False)
        
        self._create_split(metadata_csv)
        self._all_clusters = dict(enumerate(self.csv['cluster'].unique().tolist()))
        self._num_clusters = len(self._all_clusters)
    
    def _filter_metadata(self, raw_csv):
        # Filter metadata for PDB dataset
        filter_cfg = self.dataset_cfg.filter
        data_csv = raw_csv[raw_csv.oligomeric_detail.isin(filter_cfg.oligomeric)]
        data_csv = data_csv[data_csv.num_chains.isin(filter_cfg.num_chains)]
        data_csv = _length_filter(data_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
        data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
        data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
        return data_csv


class AFDBDataset(BaseDataset):
    def _filter_metadata(self, raw_csv):
        # Filter metadata for Genie2 dataset
        filter_cfg = self.dataset_cfg.filter
        data_csv = _length_filter(raw_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
        data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
        data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
        data_csv['oligomeric_detail'] = 'monomeric'
        return data_csv
    
class CATHDataset(BaseDataset):
    def __init__(self, *, dataset_cfg, is_training, task,):
        super().__init__(dataset_cfg=dataset_cfg, is_training=is_training, task=task)
        cath_list_path = dataset_cfg.domain_list_path
        self._domain_to_cath_code = self._parse_cath_domain_list(cath_list_path)
        
    def _parse_cath_domain_list(self, filepath):
        domain_to_cath = {}
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    domain_id = parts[0]
                    cath_code = f"{parts[1]}.{parts[2]}.{parts[3]}.{parts[4]}"
                    domain_to_cath[domain_id] = cath_code
        return domain_to_cath
    
    def _filter_metadata(self, raw_csv):
        # Filter metadata for CATH dataset
        filter_cfg = self.dataset_cfg.filter
        data_csv = _length_filter(raw_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
        data_csv['oligomeric_detail'] = 'monomeric'
        return data_csv
    
    def __getitem__(self, row_idx):
        feats = super().__getitem__(row_idx)

        domain_id = feats.get('pdb_name') # BaseDataset에서 넣어준 pdb_name 사용
        if domain_id:
            cath_code = self._domain_to_cath_code.get(domain_id, 'Unknown') # 조회, 없으면 'Unknown'
        else:
            cath_code = 'Unknown'
        feats['cath_label'] = cath_code # 'cath_label' 키로 추가

        return feats