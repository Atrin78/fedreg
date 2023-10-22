#!/bin/bash
#SBATCH --job-name=DMAD

#SBATCH --partition=a40

#SBATCH --gres=gpu:1

#SBATCH --qos=m3

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=8G

#SBATCH --output=/h/sayromlou/ubc/fedreg/FedAVG.out

#SBATCH --error=/h/sayromlou/ubc/fedreg/FedAVG.err

# prepare your environment here
source /ssd003/projects/aieng/public/FL_env/env3/bin/activate

# put your command here
python main.py -c tasks/mnist/FedAvg_e40_lr1/config_mine