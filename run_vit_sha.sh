#!/bin/bash -l

#SBATCH --job-name=s1sha_vit_train
#SBATCH --account=did_crowd_counting_339
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH -o logs/sha_vit_stage1_%j.out
#SBATCH -e logs/sha_vit_stage1_%j.err
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

srun python train_stage1.py --config configs/config_vit_sha.yaml --data_dir data --batch_size 16 --out checkpoints/sha/vit_b_16/stage1
#srun python trainer.py --config configs/config_vit_sha.yaml --model clip_vit_b_16  --anchor_points average --prompt_type word --dataset sha --sliding_window --window_size 224 --stride 224 --count_loss dmcount --out checkpoints/sha/vit_b_16/stage2 
#srun python trainer.py --config configs/config_vit_sha.yaml --model clip_vit_b_16  --anchor_points average --prompt_type word --dataset sha --sliding_window --window_size 224 --stride 224 --count_loss dmcount --out checkpoints/sha/vit_b_16/stage2 

#srun python train_stage3_v2.py --config configs/config_vit_sha.yaml --s1 checkpoints/sha/vit_b_16/stage1/best_model.pth --s2 checkpoints/sha/vit_b_16/stage2/best_mae_0.pth --out checkpoints/sha/vit_b_16/stage3 

#python -c "import torch; ck=torch.load('checkpoints/sha/vit_b_16/stage1/best_model.pth', map_location='cpu'); sd=ck.get('model_state_dict', ck); k=[x for x in sd if 'zip_head.shared.0.weight' in x][0]; print('key=',k,'shape=',sd[k].shape)"
