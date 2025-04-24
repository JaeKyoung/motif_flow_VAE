#!/bin/bash
#SBATCH -p gpu
#SBATCH -J vae_pf_8_0.01
#SBATCH -c 8
#SBATCH --mem=38g
#SBATCH --gres=gpu:A6000:1
#SBATCH -w gpu01
#SBATCH -o run/vae_pf_8_0.01/output.log
#SBATCH -e run/vae_pf_8_0.01/output.err

source ~/.bashrc
conda activate fm_mms_VAE

python -u -W ignore /home/worud/project/motif_flow_VAE/motifflow/experiments/train.py >> run/vae_pf_8_0.01/log.log
