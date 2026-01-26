#!/bin/bash -l

#SBATCH --job-name=s1qnrfvit_train
#SBATCH --account=did_crowd_counting_339
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH -o logs/s1_qnrf_vit_%j.out
#SBATCH -e logs/s1_qnrf_vit_%j.err
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





#VIT_B_16

#srun python train_stage1.py --config configs/config_vit_qnrf.yaml --model vit_b_16 --data_dir data --batch_size 16 --input_size 224 --reduction 8  --dataset qnrf --batch_size 16 --amp --num_crops 1 --sliding_window --window_size 224 --stride 224  --out  checkpoints/qnrf/vit_b_16/stage1_224

srun python train_stage1.py --config configs/config_vit_qnrf.yaml --dataset qnrf --model vit_b_16 --input_size 224 --reduction 16 --sliding_window --window_size 224 --stride 224 --pos_weight 2 --lr 1e-4 --lr_backbone 1e-5 --batch_size 16 --total_epochs 2000 --eval_freq 5 --amp --out checkpoints/qnrf/vit_b_16/stage1_v2


#BEST
#srun python trainer.py  --dataset qnrf  --model clip_vit_b_16  --input_size 448  --reduction 8  --truncation 4  --anchor_points average  --prompt_type word  --granularity fine  --batch_size 16  --num_crops 2  --num_workers 8  --amp  --sliding_window  --window_size 448  --stride 448  --count_loss dmcount  --weight_count_loss 1.0  --lr 1e-5  --weight_decay 1e-4  --warmup_epochs 20  --warmup_lr 1e-7  --total_epochs 2000  --eval_freq 5  --save_freq 5  --save_best_k 3 --out checkpoints/qnrf/vit_b_16/stage2_v2

#srun python trainer.py --dataset qnrf --model clip_vit_b_16 --input_size 224 --reduction 8 --truncation 4 --anchor_points average --prompt_type word --granularity fine --batch_size 16 --num_crops 2 --num_workers 8 --amp --sliding_window --window_size 224 --stride 224 --count_loss dmcount --weight_count_loss 1.0 --lr 1e-4 --weight_decay 1e-4 --warmup_epochs 50 --warmup_lr 1e-3 --total_epochs 2800 --eval_start 10 --eval_freq 10 --save_freq 5 --save_best_k 2 --out checkpoints/qnrf/vit_b_16/stage2_v2




#COME INDICATO NEL REPOUFFICIALE: 
#srun python trainer.py  --dataset qnrf  --model clip_vit_b_16  --input_size 224  --reduction 8  --truncation 4  --anchor_points average  --granularity fine  --prompt_type word  --batch_size 16  --num_workers 4  --lr 1e-4  --warmup_lr 1e-3  --weight_decay 1e-4  --count_loss dmcount  --weight_count_loss 1.0  --amp  --num_crops 2  --sliding_window  --window_size 224  --stride 224  --out checkpoints/qnrf/vit_b16/stage2_official  --num_vpt 32  --save_best_k 1

#srun python train_stage3_v2.py --config configs/config_vit_nwpu.yaml --s1 checkpoints/nwpu/vit_b_16/stage1/best_model.pth --s2 checkpoints/nwpu/vit_b_16/stage2/best_mae_0.pth --input_size 448 --sliding_window --window_size 448 --stride 448  --out checkpoints/nwpu/vit_b_16/stage3 
#python train_stage3_v2.py  --config configs/config_resnet_qnrf.yaml  --dataset qnrf  --model clip_resnet50  --s1 checkpoints/qnrf/resnet50/stage1/best_model.pth  --s2 checkpoints/qnrf/resnet50/stage2/best_mae_0.pth  --input_size 448  --batch_size 8  --lr 1e-6  --total_epochs 600  --eval_freq 1  --save_freq 10  --out checkpoints/qnrf/resnet50/stage3
#CON QUESTO AVVI STAGE3 NWPU VIT


