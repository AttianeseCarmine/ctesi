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

#srun python train_stage1.py --config configs/config_vit_shb.yaml --data_dir data --batch_size 16 --out checkpoints/shb/vit_b_16/stage1


#srun python trainer.py  --dataset shb  --model clip_vit_b_16  --input_size 448  --reduction 8  --truncation 4  --anchor_points average  --prompt_type word  --batch_size 16  --num_crops 2  --amp  --sliding_window  --window_size 448  --stride 448  --count_loss dmcount  --weight_count_loss 1.0  --out checkpoints/shb/vit_b_16/stage2

#srun python trainer.py --dataset shb --model clip_vit_b_16 --input_size 448 --reduction 8 --truncation 4 --anchor_points average --prompt_type word --batch_size 16 --num_crops 2 --amp --sliding_window --window_size 448 --stride 448 --count_loss dmcount --weight_count_loss 1.0 --out checkpoints/shb/vit_b_16/stage2

#srun python train_stage3_v2.py --config configs/config_vit_shb.yaml --s1 checkpoints/shb/vit_b_16/stage1/best_model.pth --s2 checkpoints/shb/vit_b_16/stage2/best_mae_0.pth --out checkpoints/shb/vit_b_16/stage3 


#QUESTO LO USI PER ESEGUIRE IL TRAIN STAGE3 VIT SHB
srun python train_stage3_v2.py --config configs/config_vit_shb.yaml --s1 checkpoints/shb/vit_b_16/stage1/best_model.pth --s2 checkpoints/shb/vit_b_16/stage2/best_mae_0.pth --input_size 448 --sliding_window --window_size 448 --stride 448 --out checkpoints/shb/vit_b_16/stage3 
