#!/bin/bash -l

#SBATCH --job-name=eval_stage1
#SBATCH --account=did_crowd_counting_339
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH -o logs/stage1_%j.out
#SBATCH -e logs/stage1_%j.err
#SBATCH --mail-user=c.attianese13@studenti.unisa.it
#SBATCH --mail-type=ALL

mkdir -p logs

# init modules
if ! command -v module &>/dev/null; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
  [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
fi

module purge
module load slurm/slurm/23.11.10
module load anaconda/3
module load cuda12.8/toolkit/12.8.0
module load cuda12.8/blas/12.8.0
module load cuda12.8/fft/12.8.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctesi

echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda)"
nvidia-smi

set -e
OUT_DIR="results"
mkdir -p "$OUT_DIR/stage1/confusion_matrix"
mkdir -p "$OUT_DIR/stage1/visualization"
mkdir -p "$OUT_DIR/stage1/heatmap"

# CALCOLO THRESHOLD OTTIMALE E MATRICE DI CONFUSIONE per lo stage1.
# CONFERMATI
  srun python eval_threshold.py --config checkpoints/sha/resnet50/stage1_v3/config_stage1.yaml --ckpt checkpoints/sha/resnet50/stage1_v3/best_model.pth --cm_out "$OUT_DIR/stage1/confusion_matrix/sha_resnet50.png" --backbone resnet50
# CONFERMATI
  srun python eval_threshold.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/shb/resnet50/stage1/best_model.pth --cm_out "$OUT_DIR/stage1/confusion_matrix/shb_resnet50.png" --backbone resnet50
# CONFERMATI
  srun python eval_threshold.py --config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml --ckpt checkpoints/qnrf/resnet50/stage1_v5/best_model.pth --cm_out "$OUT_DIR/stage1/confusion_matrix/qnrf_resnet50.png" --backbone resnet50
# CONFERMATI
  srun python eval_threshold.py --config checkpoints/sha/vit_b_16/stage1_v8/config_stage1.yaml --ckpt checkpoints/sha/vit_b_16/stage1_v8/best_model.pth --cm_out "$OUT_DIR/stage1/confusion_matrix/sha_vit_b_16_6.png"   --backbone vit_b_16
# CONFERMATI
  srun python eval_threshold.py --config checkpoints/shb/vit_b_16/stage1_v7_stable/config_stage1.yaml --ckpt checkpoints/shb/vit_b_16/stage1_v7_stable/best_model.pth --cm_out "$OUT_DIR/stage1/confusion_matrix/shb_vit_b_16.png" --backbone vit_b_16
# CONFERMATI
  srun python eval_threshold.py --config checkpoints/qnrf/vit_b_16/stage1_v7_stable/config_stage1.yaml --ckpt checkpoints/qnrf/vit_b_16/stage1_v7_stable/best_model.pth --cm_out "$OUT_DIR/stage1/confusion_matrix/qnrf_vit_b_16.png" --backbone vit_b_16

#
## STAGE 1 VISUALIZZAZIONe THRESHOLD CURVES
## CONFERMATI
  srun python stage1_plot_threshold_curves.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/sha/resnet50/stage1/best_model.pth --backbone resnet50 --out_dir "$OUT_DIR/stage1/sha_resnet50_threshold_curve.png"
# CONFERMATI
  srun python stage1_plot_threshold_curves.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/shb/resnet50/stage1/best_model.pth --backbone resnet50 --out_dir "$OUT_DIR/stage1/shb_resnet50_threshold_curve.png"  
# CONFERMATI
  srun python stage1_plot_threshold_curves.py --config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml --ckpt checkpoints/qnrf/resnet50/stage1_v5/best_model.pth --backbone resnet50 --out_dir "$OUT_DIR/stage1/qnrf_resnet50_threshold_curve.png" 
# CONFERMATI
  srun python stage1_plot_threshold_curves.py --config checkpoints/sha/vit_b_16/stage1_v8/config_stage1.yaml --ckpt checkpoints/sha/vit_b_16/stage1_v8/best_model.pth --backbone vit_b_16 --out_dir "$OUT_DIR/stage1/sha_vit_b_16_threshold_curve.png" 
# CONFERMATI
  srun python stage1_plot_threshold_curves.py --config checkpoints/shb/vit_b_16/stage1_v7_stable/config_stage1.yaml --ckpt checkpoints/shb/vit_b_16/stage1_v7_stable/best_model.pth --backbone vit_b_16 --out_dir "$OUT_DIR/stage1/shb_vit_b_16_threshold_curve.png" 
# CONFERMATI
  srun python stage1_plot_threshold_curves.py --config checkpoints/qnrf/vit_b_16/stage1_v7_stable/config_stage1.yaml --ckpt checkpoints/qnrf/vit_b_16/stage1_v7_stable/best_model.pth --backbone vit_b_16 --out_dir "$OUT_DIR/stage1/qnrf_vit_b_16_threshold_curve.png" 

# VISUALIZZAZIONE E HEATMAP PER IMMAGINI DI VALIDAZIONE STAGE 1 vit_b_16
srun python stage1_visualize.py --config checkpoints/sha/vit_b_16/stage1_v8/config_stage1.yaml --checkpoint checkpoints/sha/vit_b_16/stage1_v8/best_model.pth --backbone vit_b_16 --image_path data/sha/val/images/050.jpg --threshold 0.35 --out_dir "$OUT_DIR/stage1/visualization"
srun python stage1_heatmap.py --config checkpoints/sha/vit_b_16/stage1_v8/config_stage1.yaml --checkpoint checkpoints/sha/vit_b_16/stage1_v8/best_model.pth --backbone vit_b_16 --image_path data/sha/val/images/050.jpg --out_dir "$OUT_DIR/stage1/heatmap"
srun python stage1_visualize.py --config checkpoints/shb/vit_b_16/stage1_v7_stable/config_stage1.yaml --checkpoint checkpoints/shb/vit_b_16/stage1_v7_stable/best_model.pth --backbone vit_b_16 --image_path data/shb/val/images/050.jpg --threshold 0.20 --out_dir "$OUT_DIR/stage1/visualization"
srun python stage1_heatmap.py --config checkpoints/shb/vit_b_16/stage1_v7_stable/config_stage1.yaml --checkpoint checkpoints/shb/vit_b_16/stage1_v7_stable/best_model.pth --backbone vit_b_16 --image_path data/shb/val/images/050.jpg --out_dir "$OUT_DIR/stage1/heatmap"
srun python stage1_visualize.py --config checkpoints/qnrf/vit_b_16/stage1_v7_stable/config_stage1.yaml --checkpoint checkpoints/qnrf/vit_b_16/stage1_v7_stable/best_model.pth --backbone vit_b_16 --image_path data/qnrf/val/images/050.jpg --threshold 0.20 --out_dir "$OUT_DIR/stage1/visualization"
srun python stage1_heatmap.py --config checkpoints/qnrf/vit_b_16/stage1_v7_stable/config_stage1.yaml --checkpoint checkpoints/qnrf/vit_b_16/stage1_v7_stable/best_model.pth --backbone vit_b_16 --image_path data/qnrf/val/images/050.jpg  --out_dir "$OUT_DIR/stage1/heatmap"

# VISUALIZZAZIONE E HEATMAP PER IMMAGINI DI VALIDAZIONE STAGE 1 resnet50
srun python stage1_visualize.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/sha/resnet50/stage1/best_model.pth --backbone resnet50 --image_path data/sha/val/images/050.jpg --threshold 0.35 --out_dir "$OUT_DIR/stage1/visualization"
srun python stage1_heatmap.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/sha/resnet50/stage1/best_model.pth --backbone resnet50 --image_path data/sha/val/images/050.jpg --out_dir "$OUT_DIR/stage1/heatmap"
srun python stage1_visualize.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/shb/resnet50/stage1/best_model.pth --backbone resnet50 --image_path data/shb/val/images/050.jpg --threshold 0.35 --out_dir "$OUT_DIR/stage1/visualization"
srun python stage1_heatmap.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/shb/resnet50/stage1/best_model.pth --backbone resnet50 --image_path data/shb/val/images/050.jpg --out_dir "$OUT_DIR/stage1/heatmap"
srun python stage1_visualize.py --config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml --checkpoint checkpoints/qnrf/resnet50/stage1_v5/best_model.pth --backbone resnet50 --image_path data/qnrf/val/images/050.jpg --threshold 0.35 --out_dir "$OUT_DIR/stage1/visualization"
srun python stage1_heatmap.py --config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml --checkpoint checkpoints/qnrf/resnet50/stage1_v5/best_model.pth --backbone resnet50 --image_path data/qnrf/val/images/050.jpg --out_dir "$OUT_DIR/stage1/heatmap"

# -------------------------------------------------------------------
#  *** STAGE 2 ***

# STAGE 2 VISUALIZZAZIONE 
python visualize_stage2.py --config checkpoints/qnrf/resnet50/stage2_official/config.yaml --checkpoint checkpoints/qnrf/resnet50/stage2_official/best_mae_0.pth --dataset qnrf --model clip_resnet50 --out_png "$OUT_DIR/stage2/qnrf_resnet50.png" --indices "127,44,55"
python visualize_stage2.py --config checkpoints/sha/resnet50/stage2/config_stage2.yaml --checkpoint checkpoints/sha/resnet50/stage2/best_mae_0.pth --dataset sha --model clip_resnet50 --out_png "$OUT_DIR/stage2/sha_resnet50.png" --indices "10,93,153"
python visualize_stage2.py --config checkpoints/shb/resnet50/stage2/config_stage2.yaml --checkpoint checkpoints/shb/resnet50/stage2/best_mae_0.pth --dataset shb --model clip_resnet50 --out_png "$OUT_DIR/stage2/shb_resnet50.png" --indices "20,161,267"

python visualize_stage2.py --config checkpoints/sha/vit_b_16/stage2_size224_56/config.yaml --checkpoint checkpoints/sha/vit_b_16/stage2_size224_56/best_mae_0.pth --dataset sha --model clip_vit_b_16 --out_png "$OUT_DIR/stage2/sha_vit_b_16.png" --indices "10,93,153"
python visualize_stage2.py --config checkpoints/shb/vit_b_16/stage2_7/config_stage2.yaml --checkpoint checkpoints/shb/vit_b_16/stage2_7/best_mae_0.pth --dataset shb --model clip_vit_b_16 --out_png "$OUT_DIR/stage2/shb_vit_b_16.png" --indices "200,161,267"
python visualize_stage2.py --config checkpoints/qnrf/vit_b_16/stage2_v2/config_stage2.yaml --checkpoint checkpoints/qnrf/vit_b_16/stage2_v2/best_mae_0.pth --dataset qnrf --model clip_vit_b_16 --out_png "$OUT_DIR/stage2/qnrf_vit_b_16.png" --indices "20,44,55"

#STAGE3
#python evaluation_stage3.py --config_zip checkpoints/shb/vit_b_16/stage1_v2/config_stage1.yaml --config_clip checkpoints/shb/vit_b_16/stage2_v2/config.yaml --ckpt_zip checkpoints/shb/vit_b_16/stage1_v2/best_model.pth --ckpt_clip checkpoints/shb/vit_b_16/stage2_v2/best_mae_0.pth --dataset_root shb 
#python evaluation_stage3.py --config_zip --config checkpoints/sha/vit_b_16/stage1_v8/config_stage1.yaml --config_clip checkpoints/sha/vit_b_16/stage2_size224_56/config.yaml --ckpt_zip checkpoints/sha/vit_b_16/stage1_v8/best_model.pth --ckpt_clip checkpoints/sha/vit_b_16/stage2_size224_56/best_mae_0.pth --dataset_root sha



# VISUALIZZAZIONE STAGE 3 CON VIT B_16
python visualize_stage123_gating.py  --s1_config checkpoints/sha/resnet50/stage1/config_stage1.yaml  --s1_ckpt   checkpoints/sha/resnet50/stage1/best_model.pth  --s2_config checkpoints/sha/resnet50/stage2/config_stage2.yaml  --s2_ckpt  checkpoints/sha/resnet50/stage2/best_mae_0.pth  --s3_config checkpoints/sha/resnet50/stage3_OLD/config.yaml  --s3_ckpt   checkpoints/sha/resnet50/stage3_OLD/best_model.pth  --dataset sha --model clip_resnet50  --indices "17,22,55"  --mask_threshold 0.30  --out_png "$OUT_DIR/stage3/stage123_sha_resnet50.png"
python visualize_stage123_gating.py  --s1_config checkpoints/shb/resnet50/stage1/config_stage1.yaml  --s1_ckpt   checkpoints/shb/resnet50/stage1/best_model.pth  --s2_config checkpoints/shb/vit_b_16/stage2_7/config_stage2.yaml  --s2_ckpt  checkpoints/shb/vit_b_16/stage2_7/best_mae_0.pth  --s3_config checkpoints/shb/resnet50/stage3_OLD/config.yaml  --s3_ckpt   checkpoints/shb/resnet50/stage3/best_model.pth  --dataset shb --model clip_resnet50  --indices "1,22,55"  --mask_threshold 0.30  --out_png "$OUT_DIR/stage3/stage123_shb_resnet50.png"
python visualize_stage123_gating.py  --s1_config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml  --s1_ckpt   checkpoints/qnrf/resnet50/stage1_v5/best_model.pth  --s2_config checkpoints/qnrf/resnet50/stage2_official/best_mae_0.pth  --s2_ckpt  checkpoints/shb/vit_b_16/stage2_7/best_mae_0.pth  --s3_config checkpoints/shb/resnet50/stage3_OLD/config.yaml  --s3_ckpt   checkpoints/shb/resnet50/stage3/best_model.pth  --dataset shb --model clip_resnet50  --indices "1,22,55"  --mask_threshold 0.30  --out_png "$OUT_DIR/stage3/stage123_shb_resnet50.png"


python visualize_stage123_gating.py  --s1_config checkpoints/sha/vit_b_16/stage1_v8/config_stage1.yaml  --s1_ckpt   checkpoints/sha/vit_b_16/stage1_v8/best_model.pth  --s2_config checkpoints/sha/vit_b_16/stage2_size224_56/config.yaml  --s2_ckpt   checkpoints/sha/vit_b_16/stage2_size224_56/best_mae_0.pth  --s3_config checkpoints/sha/vit_b_16/stage3_v7/config.yaml  --s3_ckpt   checkpoints/sha/vit_b_16/stage3_v7_zipw0.01/best_model.pth  --dataset sha --model clip_vit_b_16  --indices "30,12,45"  --mask_threshold 0.30  --out_png "$OUT_DIR/stage3/stage123_sha_vit_b_16.png"
python visualize_stage123_gating.py  --s1_config checkpoints/shb/vit_b_16/stage1_v7_stable/config_stage1.yaml  --s1_ckpt  checkpoints/shb/vit_b_16/stage1_v7_stable/best_model.pth  --s2_config checkpoints/shb/vit_b_16/stage2_7/config_stage2.yaml  --s2_ckpt  checkpoints/shb/vit_b_16/stage2_7/best_mae_0.pth --s3_config checkpoints/shb/vit_b_16/stage3_aligned_zipw0.01/config.yaml  --s3_ckpt   checkpoints/shb/vit_b_16/stage3_aligned_zipw0.01/best_model.pth  --dataset shb --model clip_vit_b_16  --indices "7,20,12"  --mask_threshold 0.30  --out_png "$OUT_DIR/stage3/stage123_shb_vit_b_16.png"
python visualize_stage123_gating.py  --s1_config checkpoints/qnrf/vit_b_16/stage1_v7_stable/config_stage1.yaml  --s1_ckpt  checkpoints/qnrf/vit_b_16/stage1_v7_stable/best_model.pth  --s2_config  checkpoints/qnrf/vit_b_16/stage2_v2/config.yaml --s2_ckpt  checkpoints/qnrf/vit_b_16/stage2_v2/best_mae_0.pth --s3_config checkpoints/qnrf/vit_b_16/stage3_zipw0.01/config.yaml  --s3_ckpt   checkpoints/qnrf/vit_b_16/stage3_zipw0.01/best_model.pth  --dataset qnrf --model clip_vit_b_16  --indices "22,14,70"  --mask_threshold 0.25  --out_png "$OUT_DIR/stage3/stage123_qnrf_vit_b_16.png"

