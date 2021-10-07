#!/bin/bash


#SBATCH -J GD_OC
#SBATCH -o out.jiye.%j
#SBATCH -p gpu15
#SBATCH -t 11-00:00:00
#SBATCH -N 1

#SBATCH -n 2
#SBATCH --gres=gpu:1
/home/jiye/.conda/envs/jiyekim/bin/python -u ../envs/refinement.py --tag "inv_OC_refinement" --env "Inv-v1" --gpu "2" --samples 1 --preset "./RNN_OC_inv_1007.mat"