#!/bin/bash
#SBATCH --job-name=DMAD

#SBATCH --partition=a40

#SBATCH --gres=gpu:1

#SBATCH --qos=m2

#SBATCH --time=8:00:00

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=8G

# prepare your environment here
source /ssd003/projects/aieng/public/FL_env/env3/bin/activate

# put your command here
python main.py -c tasks_mine/cifar10/FedAvg_e30_lr05/config_pretrain