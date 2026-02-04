#!/bin/bash -l

#SBATCH --job-name=s3shb_vit
#SBATCH --account=did_crowd_counting_339
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH -o logs/s3_shb_vit_%j.out
#SBATCH -e logs/s3_shb_vit_%j.err
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

#srun python train_stage1.py --config configs/config_vit_shb.yaml --data_dir data --batch_size 16 --out checkpoints/shb/vit_b_16/stage1_v2

#srun python train_stage1.py --config configs/config_vit_shb.yaml --input_size 224 --reduction 8 --pos_weight 3.0 --lr_backbone 1e-5  --target_recall 0.90 --save_freq 5 --out checkpoints/shb/vit_b_16/stage1_v5/
#srun python train_stage1.py   --config configs/config_vit_shb.yaml   --input_size 224   --reduction 8   --focal_alpha 0.75   --focal_gamma 2.0   --target_recall 0.90   --lr_backbone 1e-5   --save_freq 5   --out checkpoints/shb/vit_b_16/stage1_focal_v1
#srun python trainer.py  --dataset shb  --model clip_vit_b_16  --input_size 448  --reduction 8  --truncation 4  --anchor_points average  --prompt_type word  --batch_size 16  --num_crops 2  --amp  --sliding_window  --window_size 448  --stride 448  --count_loss dmcount  --weight_count_loss 1.0 --out checkpoints/shb/vit_b_16/stage2_v2

#srun python trainer.py --dataset shb --model clip_vit_b_16 --input_size 448 --reduction 8 --truncation 4 --anchor_points average --prompt_type word --batch_size 16 --num_crops 2 --amp --sliding_window --window_size 448 --stride 448 --count_loss dmcount --weight_count_loss 1.0 --resume checkpoints/shb/vit_b_16/stage2_v2/ckpt.pth --out checkpoints/shb/vit_b_16/stage2_v2
#QUESTO LO USI PER ESEGUIRE IL TRAIN STAGE3 VIT SHB
#srun python train_stage3_v2.py --config configs/config_vit_shb.yaml --s1 checkpoints/shb/vit_b_16/stage1/best_model.pth --s2 checkpoints/shb/vit_b_16/stage2/best_mae_0.pth --input_size 448 --sliding_window --window_size 448 --stride 448 --out checkpoints/shb/vit_b_16/stage3 
# Nota: Usa una cartella --out NUOVA per evitare di caricare i vecchi checkpoint sbagliati
#srun python train_stage3_v2.py  --config configs/config_vit_shb.yaml  --s1 checkpoints/shb/vit_b_16/stage1_v2/best_model.pth  --s2 checkpoints/shb/vit_b_16/stage2_v2/best_mae_0.pth  --input_size 448  --lr 1e-4  --batch_size 4  --zip_w 0.5  --clip_w 0.1  --cons_w 20.0  --total_epochs 300  --eval_freq 1  --out ./checkpoints/shb/clip_vit_b_16/stage3_fixed_init

#srun python train_stage3_v2.py --config_stage1 configs/config_vit_shb.yaml --config_stage2 checkpoints/shb/vit_b_16/stage2_v2/config.yaml --s1 checkpoints/shb/vit_b_16/stage1_v2/best_model.pth --s2 checkpoints/shb/vit_b_16/stage2_v2/best_mae_0.pth --lr 1e-6 --weight_decay 1e-4 --total_epochs 600 --eval_freq 5 --eval_start 1 --lambda_zip 1.0 --lambda_clip 1.0 --lambda_count 10.0 --zip_pos_weight 15.0 --amp --out checkpoints/shb/vit_b_16/stage3_joint


#srun python train_stage3_v3.py --config configs/config_vit_shb.yaml --dataset shb --c2 checkpoints/shb/vit_b_16/stage2_v2/config.yaml --s1 checkpoints/shb/vit_b_16/stage1_v2/best_model.pth --s2 checkpoints/shb/vit_b_16/stage2_v2/best_mae_0.pth --out checkpoints/shb/vit_b_16/stage3_v2 --gpu 0 --input_size 448

#srun python train_stage3_v2.py  --dataset shb --config configs/config_vit_shb.yaml --s1 checkpoints/shb/vit_b_16/stage1_v2/best_model.pth --s2 checkpoints/shb/vit_b_16/stage2_v2/best_mae_0.pth --input_size 448 --sliding_window --window_size 448 --stride 448 --out checkpoints/shb/vut_b_16/stage3_v2

#srun python train_stage3_v2.py --config configs/config_vit_shb.yaml --s1 checkpoints/shb/vit_b_16/stage1_v2/best_model.pth --s2 checkpoints/shb/vit_b_16/stage2_v2/best_mae_0.pth --input_size 448 --sliding_window --window_size 448 --stride 448 --out checkpoints/shb/vit_b_16/stage3_v2_sliding --sliding_window


#srun python train_stage3_v3.py --config configs/config_vit_shb.yaml --model clip_vit_b_16 --s1 checkpoints/shb/vit_b_16/stage1_v2/best_model.pth --s2 checkpoints/shb/vit_b_16/stage2_v2/best_mae_0.pth --dataset shb --input_size 448 --sliding_window --window_size 448 --stride 448 --out checkpoints/shb/vit_b_16/stage3_v2_sliding




#srun python train_stage3_v2.py   --config configs/config_vit_shb.yaml   --model clip_vit_b_16   --s1 checkpoints/shb/vit_b_16/stage1_v5/best_model.pth   --s2 checkpoints/shb/vit_b_16/stage2_7/best_mae_0.pth   --dataset shb   --input_size 224   --reduction 8   --lr 1e-6   --sliding_window --window_size 224 --stride 224   --out checkpoints/shb/vit_b_16/stage3_v5

srun python train_stage3_clip_vit_b_16.py --config_s1 checkpoints/shb/vit_b_16/stage1_v5/config_stage1.yaml --config_s2 checkpoints/shb/vit_b_16/stage2_7/config.yaml --s1 checkpoints/shb/vit_b_16/stage1_v5/best_model.pth --s2 checkpoints/shb/vit_b_16/stage2_7/best_mae_0.pth --dataset shb --model clip_vit_b_16  --input_size 224  --reduction 8  --batch_size 4  --lr 1e-5  --max_steepness 4.0  --zip_w 0.001  --total_epochs 3500  --eval_freq 5  --out checkpoints/shb/vit_b_16/stage3_v