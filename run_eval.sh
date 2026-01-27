#!/bin/bash -l

#SBATCH --job-name=eval_nwpu_vit
#SBATCH --account=did_crowd_counting_339
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH -o logs/eval_nwpu_vit_%j.out
#SBATCH -e logs/eval_nwpu_vit_%j.err
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

#EVAL THRESHOLD STAGE1
#srun python eval_threshold.py --config checkpoints/sha/vit_b_16/stage1_v2_fp_safe/config_stage1.yaml --ckpt checkpoints/sha/vit_b_16/stage1_v2_fp_safe/best_model.pth 
#srun python eval_threshold.py --config checkpoints/sha/vit_b_16/stage1_size224/config_stage1.yaml --ckpt checkpoints/sha/vit_b_16/stage1_size224/best_model.pth 
#srun python eval_threshold.py --config checkpoints/sha/vit_b_16/stage1/config_stage1.yaml --ckpt checkpoints/sha/vit_b_16/stage1/best_model.pth 
#srun python eval_threshold.py --config checkpoints/qnrf/vit_b_16/stage1_v2/config_stage1.yaml --ckpt checkpoints/qnrf/vit_b_16/stage1_v2/best_model.pth 
srun python eval_threshold.py --config checkpoints/nwpu/vit_b_16/stage1_v3/config_stage1.yaml --ckpt checkpoints/nwpu/vit_b_16/stage1_v3/best_model.pth 

#srun python eval_threshold.py --config checkpoints/nwpu/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/nwpu/resnet50/stage1/best_model.pth 
#srun python eval_threshold.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/sha/resnet50/stage1/best_model.pth 
#srun python eval_threshold.py --config checkpoints/qnrf/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/qnrf/resnet50/stage1/best_model.pth 
#srun python eval_threshold.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/shb/resnet50/stage1/best_model.pth 




#srun python stage1_visualize.py --config checkpoints/sha/vit_b_16/stage1/config_stage1.yaml --checkpoint checkpoints/sha/vit_b_16/stage1/best_model.pth --image_path data/sha/val/images/050.jpg --threshold 0.85
#srun python stage1_heatmap.py --config checkpoints/sha/vit_b_16/stage1/config_stage1.yaml --checkpoint checkpoints/sha/vit_b_16/stage1/best_model.pth --image_path data/sha/val/images/050.jpg --threshold 0.5



#srun python stage1_visualize.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/sha/resnet50/stage1/best_model.pth --image_path data/sha/val/images/050.jpg --threshold 0.35
#srun python stage1_heatmap.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/sha/resnet50/stage1/best_model.pth --image_path data/sha/val/images/050.jpg --threshold 0.3


#srun python stage1_visualize.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/shb/resnet50/stage1/best_model.pth --image_path data/shb/val/images/050.jpg --threshold 0.35
#srun python stage1_heatmap.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/shb/resnet50/stage1/best_model.pth --image_path data/shb/val/images/050.jpg --threshold 0.3

# STAGE 2 VISUALIZZAZIONE 
#srun python visualize_stage2.py --config checkpoints/sha/clip_resnet50_sha/stage2/config_stage2.yaml --checkpoint checkpoints/sha/clip_resnet50_sha/stage2/best_mae_0.pth --image_path data/sha/val/images/050.jpg  
#python visualize_stage2.py --config checkpoints/shb/vit_b_16/stage2/config_stage2.yaml --checkpoint checkpoints/shb/vit_b_16/stage2/best_mae_0.pth --dataset shb --model clip_vit_b_16 --out_png shb_vit_b_16.png
#python visualize_stage2.py --config checkpoints/shb/resnet50/stage2/config_stage2.yaml --checkpoint checkpoints/shb/resnet50/stage2/best_mae_0.pth --dataset shb --model clip_resnet50 --out_png shb_resnet50.png
#python visualize_stage2.py --config checkpoints/qnrf/vit_b_16/stage2/config_stage2.yaml --checkpoint checkpoints/qnrf/vit_b_16/stage2/best_mae_0.pth --dataset qnrf --model clip_vit_b_16 --out_png qnrf_vit_b_16.png
#python visualize_stage2.py --config checkpoints/qnrf/resnet50/stage2/config_stage2.yaml --checkpoint checkpoints/qnrf/resnet50/stage2/best_mae_0.pth --dataset qnrf --model clip_resnet50 --out_png qnrf_resnet50.png


#STAGE3

#python evaluation_stage3.py --config_zip checkpoints/shb/vit_b_16/stage1_v2/config_stage1.yaml --config_clip checkpoints/shb/vit_b_16/stage2_v2/config.yaml --ckpt_zip checkpoints/shb/vit_b_16/stage1_v2/best_model.pth --ckpt_clip checkpoints/shb/vit_b_16/stage2_v2/best_mae_0.pth --dataset_root shb 



