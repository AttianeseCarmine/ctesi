#!/bin/bash -l

#SBATCH --job-name=S2resNWPU_train
#SBATCH --account=did_crowd_counting_339
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH -o logs/s2_nwpu_resnet_%j.out
#SBATCH -e logs/s2_nwpu_resnet_%j.err
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

#RESNET
#srun python train_stage1.py  --config configs/config_resnet_nwpu.yaml  --dataset nwpu  --model resnet50  --data_dir data  --batch_size 16  --pos_weight 10  --sliding_window  --window_size 448  --stride 448  --out checkpoints/nwpu/resnet50/stage1
  
srun python trainer.py --dataset nwpu --model clip_resnet50 --input_size 448 --reduction 8 --truncation 4 --granularity fine --prompt_type word --batch_size 16 --lr 1e-4  --weight_decay 1e-4 --warmup_epochs 10 --warmup_lr 1e-6 --count_loss dmcount --weight_count_loss 1.0 --amp --num_crops 2 --sliding_window --window_size 448 --stride 448 --out checkpoints/nwpu/resnet50/stage2



#srun python train_stage1.py --config configs/config_resnet_nwpu.yaml --dataset nwpu --data_dir data --batch_size 16 --out checkpoints/nwpu/resnet50/stage1 --sliding_window --window_size 448 --stride 448
#srun python trainer.py --dataset nwpu  --model clip_resnet50  --input_size 448  --reduction 8  --truncation 4  --granularity fine  --prompt_type word  --batch_size 16  --lr 1e-4  --weight_decay 1e-4  --warmup_epochs 10  --warmup_lr 1e-6  --count_loss dmcount  --weight_count_loss 1.0  --amp  --num_crops 2  --sliding_window  --window_size 448  --stride 448  --out checkpoints/nwpu/resnet50/stage2
#srun python trainer.py --dataset qnrf  --model clip_resnet50  --input_size 448  --reduction 8  --truncation 4  --granularity fine  --prompt_type word  --batch_size 16  --lr 1e-4  --weight_decay 1e-4  --warmup_epochs 10  --warmup_lr 1e-6  --count_loss dmcount  --weight_count_loss 1.0  --amp  --num_crops 2  --sliding_window  --window_size 448  --stride 448  --out checkpoints/qnrf/resnet50/stage2

#srun python trainer.py --dataset qnrf --model clip_resnet50 --input_size 448 --reduction 8 --truncation 4 --granularity fine --prompt_type word --batch_size 16 --lr 1e-4 --weight_decay 1e-4 --warmup_epochs 10 --warmup_lr 1e-6 --count_loss dmcount --weight_count_loss 1.0 --amp --num_crops 2 --sliding_window --window_size 448 --stride 448 --out checkpoints/qnrf/resnet50/stage2

#RESNET
#srun python train_stage1.py --config configs/config_resnet_qnrf.yaml --dataset nwpu --data_dir data --batch_size 16  --out checkpoints/nwpu/resnet50/stage1
#srun python train_stage3_v2.py --config configs/config_resnet_nwpu.yaml --s1 checkpoints/nwpu/resnet50/stage1/best_model.pth --s2 checkpoints/nwpu/resnet50/stage2/best_mae_0.pth --out checkpoints/qnrf/resnet50/stage3 

#srun python train_stage3_v2.py --config configs/config_resnet_qnrf.yaml --s1 checkpoints/qnrf/resnet50/stage1/best_model.pth --s2 checkpoints/qnrf/resnet50/stage2/best_mae_0.pth --out checkpoints/qnrf/resnet50/stage3 


#srun python train_stage3_v2.py --config configs/config_resnet_qnrf.yaml --s1 checkpoints/qnrf/resnet50/stage1/best_model.pth --s2 checkpoints/qnrf/resnet50/stage2/best_mae_0.pth --input_size 448 --sliding_window --window_size 448 --stride 448 --out checkpoints/qnrf/resnet50/stage3 
