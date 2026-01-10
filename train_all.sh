#!/bin/bash
#SBATCH --job-name=Clip_ebc_train
#SBATCH --account=did_crowd_counting_339
#SBATCH --partition=aiq 
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1
#SBATCH --time=07:00:00 
#SBATCH --output=master_log_%j.txt
#SBATCH --error=master_err_%j.txt
#SBATCH --mail-user=c.attianese13@studenti.unisa.it
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -e  # Esce immediatamente se un comando fallisce


# --- STAGE 1: π-Head Training ---
echo "▶️  STAGE 1: Training π-Head... su shb"
python train_stage1.py --config configs/config_shb.yaml
# --- STAGE 2: EBC-Head Training ---
echo "▶️  STAGE 2: Training CLIP-EBC Head..."
#python train_stage2.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage2_train.log"
python train_stage2.py --config configs/config_shb.yaml 
# --- STAGE 3: Joint Fine-tuning ---
#echo "▶️  STAGE 3: Joint Fine-tuning End-to-End..."
#python train_stage3.py --config $CONFIG_FILE 2>&1 | tee "$LOG_DIR/stage3_train.log"

python train_stage3.py --config configs/config_shb.yaml --ckpt_stage1 checkpoints/shb/stage1/best_model.pth --ckpt_stage2 checkpoints/shb/stage2/best_model.pth