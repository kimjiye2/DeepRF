#!/bin/bash



#SBATCH -J rnn_inv_v1
#SBATCH -o out.jiye.%j
#SBATCH -p gpu11
#SBATCH -t 11-00:00:00
#SBATCH -N 1

#SBATCH -n 2
#SBATCH --gres=gpu:1

RANDOM=`date "+%N"`
for i in {1..17}
do
  python time.py;

  /home/jiye/.conda/envs/jiyekim/bin/python -u ../envs/generation.py --tag "rnn_inv_OC" --env "Inv-v1" --seed $RANDOM --gpu "1"
  python time.py


done