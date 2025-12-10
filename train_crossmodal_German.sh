#!/bin/bash

# Train on German, Test on English and German

cd /data/user_data/zeyangz/HiCMAE_LSMA

# Configuration
TRAIN_LANG='German'
DATA_ROOT='/data/user_data/zeyangz/MAV-Celeb_v3'
ENCODED_ROOT='/data/user_data/zeyangz/MAV-Celeb_v3/Pre-encoded'

# Data files
TRAIN_FILE="${DATA_ROOT}/v3/German_train.txt"
TEST_FILE_ENGLISH="${DATA_ROOT}/v3/English_test.txt"
TEST_FILE_GERMAN="${DATA_ROOT}/v3/German_test.txt"

# Training parameters
lr=1e-4
epochs=100
batch_size=512
warmup_epochs=5
alpha=1.0  # Weight for contrastive loss (OPL)

# Model parameters
audio_dim=1024
image_dim=1536
hidden_dim=512
embed_dim=256
num_classes=50
dropout=0.5
activation='gelu'

# Output directory
OUTPUT_DIR="./saved/model/crossmodal_verification/train_${TRAIN_LANG}_lr${lr}_epoch${epochs}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mae

echo "=================================================="
echo "Training Cross-modal Verification on ${TRAIN_LANG}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Training file: ${TRAIN_FILE}"
echo "Encoded features: ${ENCODED_ROOT}"
echo "Learning rate: ${lr}"
echo "Epochs: ${epochs}"
echo "Batch size: ${batch_size} per GPU"
echo "=================================================="

# Train
export NCCL_P2P_DISABLE=1
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use-env \
    --master_port 13302 \
    train_crossmodal.py \
    --data_file ${TRAIN_FILE} \
    --encoded_root ${ENCODED_ROOT} \
    --data_root ${DATA_ROOT} \
    --audio_dim ${audio_dim} \
    --image_dim ${image_dim} \
    --hidden_dim ${hidden_dim} \
    --embed_dim ${embed_dim} \
    --num_classes ${num_classes} \
    --dropout ${dropout} \
    --activation ${activation} \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --lr ${lr} \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --warmup_epochs ${warmup_epochs} \
    --alpha ${alpha} \
    --num_workers 12 \
    --save_freq 5 \
    --output_dir ${OUTPUT_DIR} \
    >${OUTPUT_DIR}/nohup.out 2>&1

echo ""
echo "Training completed!"
echo ""
echo "=================================================="
echo "Testing on English and German"
echo "=================================================="

# Test on both English and German
python test_crossmodal.py \
    --train_lang ${TRAIN_LANG} \
    --checkpoint ${OUTPUT_DIR}/checkpoint_final.pth \
    --encoded_root ${ENCODED_ROOT} \
    --data_root ${DATA_ROOT} \
    --english_test ${TEST_FILE_ENGLISH} \
    --german_test ${TEST_FILE_GERMAN} \
    --audio_dim ${audio_dim} \
    --image_dim ${image_dim} \
    --hidden_dim ${hidden_dim} \
    --embed_dim ${embed_dim} \
    --num_classes ${num_classes} \
    --dropout ${dropout} \
    --activation ${activation} \
    --batch_size 128 \
    --num_workers 12 \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "Testing completed!"
echo "Results saved to:"
echo "  ${OUTPUT_DIR}/sub_score_English_unheard.txt"
echo "  ${OUTPUT_DIR}/sub_score_German_heard.txt"
echo ""

