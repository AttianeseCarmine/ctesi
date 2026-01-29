#!/bin/bash -l

#SBATCH --job-name=s3sha_vit_train
#SBATCH --account=did_crowd_counting_339
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH -o logs/s3_sha_vit_%j.out
#SBATCH -e logs/s3_sha_vit_%j.err
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



#srun python train_stage1.py --config configs/config_vit_sha.yaml --dataset sha --model vit_b_16 --input_size 224 --reduction 16 --sliding_window --window_size 224 --stride 224 --pos_weight 1.0 --lr 1e-4 --lr_backbone 1e-5 --total_epochs 2000 --eval_freq 5 --out checkpoints/sha/vit_b_16/stage1_v2_fp_safe


#srun python trainer.py  --dataset sha  --model clip_vit_b_16  --input_size 224  --reduction 8  --truncation 4  --anchor_points average  --prompt_type word  --batch_size 16  --num_crops 2  --amp  --sliding_window  --window_size 224  --stride 224  --warmup_lr 1e-3  --count_loss dmcount  --weight_count_loss 1.0  --out checkpoints/sha/vit_b_16/stage2_v2
#srun python trainer.py --dataset sha --model clip_vit_b_16 --input_size 224 --reduction 8 --truncation 4 --anchor_points average --prompt_type word --batch_size 16 --num_crops 2 --amp --sliding_window --window_size 224 --stride 224 --count_loss dmcount --weight_count_loss 1.0 --out checkpoints/sha/vit_b_16/stage2 --resume checkpoints/sha/vit_b_16/stage2_v2/best_mae_0.pth

#srun python train_stage3_v2.py --config configs/config_vit_sha.yaml --model clip_vit_b_16 --dataset sha --s1 checkpoints/sha/vit_b_16/stage1_size224/best_model.pth --s2 checkpoints/sha/vit_b_16/stage2_v2/best_mae_0.pth --out checkpoints/sha/resnet50/stage3_v2 --lr 1e-4 --total_epochs 300 --base_steepness 1.0 --max_steepness 5.0 --zip_w 1.0 --cons_w 0.05



#python -c "import torch; ck=torch.load('checkpoints/sha/vit_b_16/stage1/best_model.pth', map_location='cpu'); sd=ck.get('model_state_dict', ck); k=[x for x in sd if 'zip_head.shared.0.weight' in x][0]; print('key=',k,'shape=',sd[k].shape)"


#QUESTO LO USI PER TRAIN STAGE3 SHA
#srun python train_stage3_v2.py --config configs/config_vit_sha.yaml --s1 checkpoints/sha/vit_b_16/stage1/best_model.pth --s2 checkpoints/sha/vit_b_16/stage2/best_mae_0.pth --input_size 448 --sliding_window --window_size 448 --stride 448  --out checkpoints/sha/vit_b_16/stage3 

#srun python train_stage3_v2.py  --config configs/config_vit_sha.yaml  --s1 checkpoints/sha/vit_b_16/stage1_v2/best_model.pth  --s2 checkpoints/sha/vit_b_16/stage2_v2/best_mae_0.pth  --input_size 448  --lr 1e-4  --batch_size 4  --zip_w 0.5  --clip_w 0.1  --cons_w 20.0  --total_epochs 50  --eval_freq 1  --out ./checkpoints/sha/clip_vit_b_16/stage3_fixed_init




#srun python train_stage3_v2.py --config_stage1 configs/config_vit_sha.yaml --config_stage2 checkpoints/sha/vit_b_16/stage2/config.yaml --s1 checkpoints/sha/vit_b_16/stage1_v2_fp_safe/best_model.pth --s2 checkpoints/sha/vit_b_16/stage2/best_mae_0.pth --lr 1e-6 --weight_decay 1e-4 --total_epochs 600 --eval_freq 5 --eval_start 1 --lambda_zip 1.0 --lambda_clip 1.0 --lambda_count 10.0 --zip_pos_weight 15.0 --amp --out checkpoints/sha/vit_b_16/stage3_joint



srun python train_stage3_v2.py   --config configs/config_vit_sha.yaml   --model clip_vit_b_16   --s1 checkpoints/sha/vit_b_16/stage1_size224/best_model.pth   --s2 checkpoints/sha/vit_b_16/stage2_size224_56/best_mae_0.pth   --dataset sha   --input_size 224   --reduction 8   --lr 1e-6   --sliding_window --window_size 224 --stride 224   --out checkpoints/sha/vit_b_16/stage3_P9