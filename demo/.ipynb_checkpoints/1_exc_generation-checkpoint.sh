#!/bin/bash



#SBATCH -J rnn_inv_v0
#SBATCH -o out.jiye.%j
#SBATCH -p gpu14
#SBATCH -t 11-00:00:00
#SBATCH -N 1

#SBATCH -n 2
#SBATCH --gres=gpu:1

RANDOM=`date "+%N"`
for i in {1..17}
do
  python time.py;

  /home/jiye/.conda/envs/jiyekim/bin/python -u ../envs/generation.py --tag "rnn_inv_origin" --env "Inv-v0" --seed $RANDOM --gpu "3"
  python time.py


done