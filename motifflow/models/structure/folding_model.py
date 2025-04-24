import os
import esm
import subprocess
import logging
import torch
import json
import numpy as np
from biotite.sequence.io import fasta
import pandas as pd
import glob

class FoldingModel:
    
    def __init__(self, cfg, device_id=None):
        self._print_logger = logging.getLogger(__name__)
        self._cfg = cfg
        self._esmf = None
        self._device_id = device_id
        self._device = None

    @property
    def device_id(self):
        if self._device_id is None:
            self._device_id = torch.cuda.current_device()
        return self._device_id

    @property
    def device(self):
        if self._device is None:
            self._device = f'cuda:{self.device_id}'
        return self._device

    def fold_fasta(self, fasta_path, output_dir):
        if self._cfg.folding_model == 'esmf':
            folded_output = self._esmf_model(fasta_path, output_dir)
        elif self._cfg.folding_model == 'af2':
            folded_output = self._af2_model(fasta_path, output_dir)
        else:
            raise ValueError(f'Unknown folding model: {self._cfg.folding_model}')
        return folded_output

    @torch.no_grad()
    def _esmf_model(self, fasta_path, output_dir):
        if self._esmf is None:
            self._print_logger.info(f'Loading ESMFold on device {self.device}')
            torch.hub.set_dir(self._cfg.pt_hub_dir)
            self._esmf = esm.pretrained.esmfold_v1().eval().to(self.device)
        fasta_seqs = fasta.FastaFile.read(fasta_path)
        folded_outputs = {
            'folded_path': [],
            'header': [],
            'plddt': [],
            'seq': [],
            'pae': [],
        }
        for header, string in fasta_seqs.items():
            # Run ESMFold
            # Need to convert unknown amino acids to alanine since ESMFold 
            # doesn't like them and will remove them...
            string = string.replace('X', 'A')
            esmf_sample_path = os.path.join(output_dir, f'folded_{header}.pdb')
            esmf_outputs = self._esmf.infer(string)
            pdb_output = self._esmf.output_to_pdb(esmf_outputs)[0]
            with open(esmf_sample_path, "w") as f:
                f.write(pdb_output)
            mean_plddt = esmf_outputs['mean_plddt'][0].item()
            folded_outputs['folded_path'].append(esmf_sample_path)
            folded_outputs['header'].append(header)
            folded_outputs['plddt'].append(mean_plddt)
            folded_outputs['seq'].append(string)
            pae = (esmf_outputs["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)).mean(-1) * 31
            mask = esmf_outputs["atom37_atom_exists"][0,:,1] == 1
            mask = mask.cpu()
            pae = pae[mask,:][:,mask]
            mean_pae = np.mean(pae)  # PAE 행렬의 평균값 계산
            folded_outputs['pae'].append(mean_pae)
        return pd.DataFrame(folded_outputs)

    def _af2_model(self, fasta_path, output_dir):
        af2_args = [
            self._cfg.colabfold_path,
            fasta_path,
            output_dir,
            '--msa-mode',
            'single_sequence',
            '--num-models',
            '1',
            '--random-seed',
            '123',
            '--device',
            f'{self.device_id}',
            '--model-order',
            '4',
            '--num-recycle',
            '3',
            '--model-type',
            'alphafold2_ptm',
        ]
        process = subprocess.Popen(
            af2_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        _ = process.wait()
        fasta_seqs = fasta.FastaFile.read(fasta_path)
        folded_outputs = {
            'folded_path': [],
            'header': [],
            'plddt': [],
            'seq': [],
            'pae': [],
        }
        all_af2_files = glob.glob(os.path.join(output_dir, '*'))
        af2_model_4_pdbs = {}
        af2_model_4_jsons = {}
        for x in all_af2_files:
            if 'model_4' in x:
                seq_name = os.path.basename(x)
                if x.endswith('.json'):
                    seq_name = seq_name.split('_scores')[0]
                    af2_model_4_jsons[seq_name] = x
                if x.endswith('.pdb'):
                    seq_name = seq_name.split('_unrelaxed')[0]
                    af2_model_4_pdbs[seq_name] = x
            else:
                os.remove(x)
        for header, string in fasta_seqs.items():
            af2_folded_path = af2_model_4_pdbs[header]
            af2_json_path = af2_model_4_jsons[header]
            with open(af2_json_path, 'r') as f:
                folded_confidence = json.load(f)
            mean_plddt = np.mean(folded_confidence['plddt'])
            folded_outputs['folded_path'].append(af2_folded_path)
            folded_outputs['header'].append(header)
            folded_outputs['plddt'].append(mean_plddt)
            folded_outputs['seq'].append(string)
            mean_pae = np.mean(folded_confidence['predicted_aligned_error'])
            folded_outputs['pae'].append(mean_pae)
        return pd.DataFrame(folded_outputs)
    
    def run_proteinmpnn(self, input_dir, output_path, motif_mask, chain_index):
        os.makedirs(os.path.join(input_dir, 'seqs'), exist_ok=True)
        if motif_mask is not None:
            # motif-scaffolding
            path_for_fixed_positions = os.path.join(input_dir, "fixed_pdbs.jsonl")
            path_for_assigned_chains = os.path.join(input_dir, "assigned_pdbs.jsonl")
            
            # fixed_positions = ' '.join(str(i+1) for i, mask in enumerate(motif_mask.squeeze().tolist()) if mask == 1)
            # Get only Chain A (chain_index==1) part of motif_mask
            motif_mask_np = motif_mask.squeeze().cpu().numpy().flatten()
            chain_index_np = chain_index.cpu().numpy().flatten()

            # Get Chain A positions
            chain_a_mask = motif_mask_np[chain_index_np == 1]
            fixed_positions = ' '.join(str(i+1) for i, mask in enumerate(chain_a_mask) if mask == 1)
            # fixed_positions = ' '.join(str(i+1) for i, mask in enumerate(chain_a_mask.tolist()) if mask == 1)
            chain_list = 'A' # chains to design

            process = subprocess.Popen([
                'python',
                os.path.join(self._cfg.pmpnn_path,'helper_scripts/parse_multiple_chains.py'),
                f'--input_path={input_dir}',
                f'--output_path={output_path}',
            ])
            _ = process.wait()
            
            process = subprocess.Popen([
                'python',
                os.path.join(self._cfg.pmpnn_path,'helper_scripts/assign_fixed_chains.py'),
                f'--input_path={output_path}',
                f'--output_path={path_for_assigned_chains}',
                f'--chain_list={str(chain_list)}',
            ])
            _ = process.wait()

            process = subprocess.Popen([
                'python',
                os.path.join(self._cfg.pmpnn_path,'helper_scripts/make_fixed_positions_dict.py'),
                f'--input_path={output_path}',
                f'--output_path={path_for_fixed_positions}',
                f'--chain_list={str(chain_list)}',
                f'--position_list={fixed_positions}'
            ])
            _ = process.wait()
            num_tries = 0
            ret = -1

            pmpnn_args = [
                'python',
                os.path.join(self._cfg.pmpnn_path,'protein_mpnn_run.py'),
                '--out_folder',
                input_dir,
                '--jsonl_path',
                output_path,
                '--chain_id_jsonl',
                path_for_assigned_chains,
                '--fixed_positions_jsonl',
                path_for_fixed_positions,
                '--num_seq_per_target',
                str(self._cfg.seq_per_sample),
                '--sampling_temp',
                '0.1',
                '--seed',
                '37',
                '--batch_size',
                '1',
                '--device',
                str(self.device_id),
            ]
            while ret < 0:
                try:
                    process = subprocess.Popen(
                        pmpnn_args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT
                    )
                    ret = process.wait()
                except Exception as e:
                    num_tries += 1
                    self._log.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                    torch.cuda.empty_cache()
                    if num_tries > 4:
                        raise e
            _ = process.wait()
        
        else: 
            #unconditional
            process = subprocess.Popen([
                'python',
                os.path.join(self._cfg.pmpnn_path,'helper_scripts/parse_multiple_chains.py'),
                f'--input_path={input_dir}',
                f'--output_path={output_path}',
            ])
            _ = process.wait()
            num_tries = 0
            ret = -1

            pmpnn_args = [
                'python',
                os.path.join(self._cfg.pmpnn_path, 'protein_mpnn_run.py'),
                '--out_folder',
                input_dir,
                '--jsonl_path',
                output_path,
                '--num_seq_per_target',
                str(self._cfg.seq_per_sample),
                '--sampling_temp',
                '0.1',
                '--seed',
                '38',
                '--batch_size',
                '1',
                '--device',
                str(self.device_id),
            ]
            while ret < 0:
                try:
                    process = subprocess.Popen(
                        pmpnn_args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT
                    )
                    ret = process.wait()
                except Exception as e:
                    num_tries += 1
                    self._print_logger.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                    torch.cuda.empty_cache()
                    if num_tries > 4:
                        raise e
            _ = process.wait()