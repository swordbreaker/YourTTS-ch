#!/bin/bash

#SBATCH --job-name=train_YourTTS     #Name of your job
#SBATCH --cpus-per-task=4    #Number of cores to reserve
#SBATCH --mem-per-cpu=32G     #Amount of RAM/core to reserve
#SBATCH --time=4-00:00:00      #Maximum allocated time
#SBATCH --qos=1week       #Selected queue to allocate your job
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --output=log/train_yourtts.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=log/train_yourtts.o%j    #Path and name to the file for the STDERR

#export TF_FORCE_GPU_ALLOW_GROWTH=true

ml CUDA/11.7.0
ml Miniconda2/4.3.30
source activate tts

export TRAINER_TELEMETRY=0
export CUDA_VISIBLE_DEVICES=0
python train_yourtts_new.py