#!/bin/bash -l

#SBATCH --job-name=eval_stage2
#SBATCH --account=did_crowd_counting_339
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH -o logs/eval_stage2_%j.out
#SBATCH -e logs/eval_stage2_%j.err
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


srun python evaluate_stage2.py --config configs/config_resnet_sha.yaml --ckpt checkpoints/sha_res50/stage2/best_mae.pth --backbone resnet50 --split val --sliding_window --window_size 224 --stride 224 --reduction 8 
srun python evaluate_stage2.py --config configs/config_resnet_shb.yaml --ckpt checkpoints/shb_res50/stage2/best_mae.pth --backbone resnet50 --split val --sliding_window --window_size 224 --stride 224 --reduction 8 
srun python evaluate_stage2.py --config configs/config_vit_shb.yaml --ckpt checkpoints/shb_vit/stage2/best_mae.pth --backbone vit_b_16 --split val --sliding_window --window_size 224 --stride 224 --reduction 16
srun python evaluate_stage2.py --config configs/config_vit_sha.yaml --ckpt checkpoints/sha_vit/stage2/best_mae.pth --backbone vit_b_16 --split val --sliding_window --window_size 224 --stride 224 --reduction 16
