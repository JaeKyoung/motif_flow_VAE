#!/bin/bash
#SBATCH -p gpu
#SBATCH -J joint_pdb2
#SBATCH -c 8
#SBATCH --mem=38g
#SBATCH --gres=gpu:A5000:1
#SBATCH -w gpu02
#SBATCH -o ./notebook/tSNE/output.log
#SBATCH -e ./notebook/tSNE/output.err

source ~/.bashrc
conda activate fm_mms_VAE

python -u -W ignore /home/worud/project/motif_flow_VAE/notebook/tSNE/t-SNE_CATH_SS.py >> ./notebook/tSNE/log.log
