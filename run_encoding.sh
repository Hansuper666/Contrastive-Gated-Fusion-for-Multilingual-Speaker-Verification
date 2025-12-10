#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mae

# Change to the preprocess directory
cd /data/user_data/zeyangz/HiCMAE_LSMA/preprocess

# Run encoding script
python encode_mavceleb.py \
    --data_root /data/user_data/zeyangz/MAV-Celeb_v3 \
    --output_root /data/user_data/zeyangz/MAV-Celeb_v3/Pre-encoded \
    --dinov2_path /data/user_data/zeyangz/HiCMAE_LSMA/saved/model/pretrained/dinov2-giant \
    --wavlm_path /data/user_data/zeyangz/HiCMAE_LSMA/saved/model/pretrained/wavlm-large \
    --num_gpus 2 \
    --num_workers 12

