#!/bin/bash
#SBATCH -p gpu
#SBATCH -J test_joint
#SBATCH -c 8
#SBATCH --mem=38g
#SBATCH --gres=gpu:A5000:1
#SBATCH -w gpu02
#SBATCH -o ./../inference_outputs/motif_flow_VAE/v0.1.2/MPNN_4_0.01_4/output.log
#SBATCH -e ./../inference_outputs/motif_flow_VAE/v0.1.2/MPNN_4_0.01_4/output.err

source ~/.bashrc
conda activate fm_mms_VAE

# Unconditional
# python -u -W ignore /home/worud/project/motif_flow/motifflow/experiments/inference_se3_flows.py -cn inference_unconditional >> ./../inference_outputs/motif_flow/v0.1.2/multimer/multimer_benchmark_test/log.log

# Motif Scaffolding
python /home/worud/project/motif_flow_VAE/motifflow/experiments/inference_joint.py >> ./../inference_outputs/motif_flow_VAE/v0.1.2/MPNN_4_0.01_4/log.log
# inference.samples.target_subset=['1PRW']
