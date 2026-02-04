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

#EVAL THRESHOLD STAGE1
#srun python eval_threshold.py --config checkpoints/sha/vit_b_16/stage1_v2_fp_safe/config_stage1.yaml --ckpt checkpoints/sha/vit_b_16/stage1_v2_fp_safe/best_model.pth 
#srun python eval_threshold.py --config checkpoints/sha/vit_b_16/stage1_v4/config_stage1.yaml --ckpt checkpoints/sha/vit_b_16/stage1_v4/best_model.pth 
#srun python eval_threshold.py --config checkpoints/sha/vit_b_16/stage1_v4/config_stage1.yaml --ckpt checkpoints/sha/vit_b_16/stage1_v4/best_model.pth 
#srun python eval_threshold.py --config checkpoints/qnrf/vit_b_16/stage1_v4/config_stage1.yaml --ckpt checkpoints/qnrf/vit_b_16/stage1_v4/best_model.pth 
#srun python eval_threshold.py --config checkpoints/nwpu/vit_b_16/stage1_v3/config_stage1.yaml --ckpt checkpoints/nwpu/vit_b_16/stage1_v3/best_model.pth 

#srun python eval_threshold.py --config checkpoints/nwpu/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/nwpu/resnet50/stage1/best_model.pth 
#srun python eval_threshold.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/sha/resnet50/stage1/best_model.pth 
#srun python eval_threshold.py --config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml --ckpt checkpoints/qnrf/resnet50/stage1_v5/best_model.pth 
#srun python eval_threshold.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/shb/resnet50/stage1/best_model.pth 

# CALCOLO THRESHOLD OTTIMALE E MATRICE DI CONFUSIONE per lo stage1.
#srun  python eval_threshold.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/sha/resnet50/stage1/best_model.pth --cm_out stage1/sha_resnet50.png --backbone resnet50
#srun python eval_threshold.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/shb/resnet50/stage1/best_model.pth --cm_out stage1/shb_resnet50.png --backbone resnet50
#srun python eval_threshold.py --config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml --ckpt checkpoints/qnrf/resnet50/stage1_v5best_model.pth --cm_out stage1/qnrf_resnet50.png --backbone resnet50
#srun python eval_threshold.py --config checkpoints/sha/vit_b_16/stage1_v5/config_stage1.yaml --ckpt checkpoints/sha/vit_b_16/stage1_v5/best_model.pth --cm_out stage1/sha_vit_b_16.png --backbone vit_b_16
#srun python eval_threshold.py --config checkpoints/shb/vit_b_16/stage1_v5/config_stage1.yaml --ckpt checkpoints/shb/vit_b_16/stage1_v5/best_model.pth --cm_out stage1/shb_vit_b_16.png --backbone vit_b_16
#srun python eval_threshold.py --config checkpoints/qnrf/vit_b_16/stage1_v5/config_stage1.yaml --ckpt checkpoints/qnrf/vit_b_16/stage1_v5/best_model.pth --cm_out stage1/qnrf_vit_b_16.png --backbone vit_b_16
#
# STAGE 1 VISUALIZZAZIONe THRESHOLD CURVES
#srun python stage1_plot_threshold_curves.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/sha/resnet50/stage1/best_model.pth --out_dir stage1/sha_resnet50_threshold_curve.png 
#srun python stage1_plot_threshold_curves.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --ckpt checkpoints/shb/resnet50/stage1/best_model.pth --out_dir stage1/shb_resnet50_threshold_curve.png  
#srun python stage1_plot_threshold_curves.py --config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml --ckpt checkpoints/qnrf/resnet50/stage1_v5/best_model.pth --out_dir stage1/qnrf_resnet50_threshold_curve.png 
#srun python stage1_plot_threshold_curves.py --config checkpoints/sha/vit_b_16/stage1_v5/config_stage1.yaml --ckpt checkpoints/sha/vit_b_16/stage1_v5/best_model.pth --out_dir stage1/sha_vit_b_16_threshold_curve.png 
#srun python stage1_plot_threshold_curves.py --config checkpoints/shb/vit_b_16/stage1_v5/config_stage1.yaml --ckpt checkpoints/shb/vit_b_16/stage1_v5/best_model.pth --out_dir stage1/shb_vit_b_16_threshold_curve.png 
#srun python stage1_plot_threshold_curves.py --config checkpoints/qnrf/vit_b_16/stage1_v5/config_stage1.yaml --ckpt checkpoints/qnrf/vit_b_16/stage1_v5/best_model.pth --out_dir stage1/qnrf_vit_b_16_threshold_curve.png 



# VISUALIZZAZIONE E HEATMAP PER IMMAGINI DI VALIDAZIONE STAGE 1
#srun python stage1_visualize.py --config checkpoints/sha/vit_b_16/stage1/config_stage1.yaml --checkpoint checkpoints/sha/vit_b_16/stage1/best_model.pth --image_path data/sha/val/images/050.jpg --threshold 0.85
#srun python stage1_heatmap.py --config checkpoints/sha/vit_b_16/stage1/config_stage1.yaml --checkpoint checkpoints/sha/vit_b_16/stage1/best_model.pth --image_path data/sha/val/images/050.jpg --threshold 0.5



#srun python stage1_visualize.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/sha/resnet50/stage1/best_model.pth --image_path data/sha/val/images/050.jpg --threshold 0.35
#srun python stage1_heatmap.py --config checkpoints/sha/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/sha/resnet50/stage1/best_model.pth --image_path data/sha/val/images/050.jpg --threshold 0.3
#srun python stage1_heatmap.py --config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml --checkpoint checkpoints/qnrf/resnet50/stage1_v5/best_model.pth --image_path data/qnrf/val/images/050.jpg --threshold 0.3
#srun python stage1_visualize.py --config checkpoints/qnrf/resnet50/stage1_v5/config_stage1.yaml --checkpoint checkpoints/qnrf/resnet50/stage1_v5/best_model.pth --image_path data/qnrf/val/images/050.jpg --threshold 0.35


#srun python stage1_visualize.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/shb/resnet50/stage1/best_model.pth --image_path data/shb/val/images/050.jpg --threshold 0.35
#srun python stage1_heatmap.py --config checkpoints/shb/resnet50/stage1/config_stage1.yaml --checkpoint checkpoints/shb/resnet50/stage1/best_model.pth --image_path data/shb/val/images/050.jpg --threshold 0.3

# STAGE 2 VISUALIZZAZIONE 
srun python visualize_stage2.py --config checkpoints/sha/resnet50/stage2/config_stage2.yaml --checkpoint checkpoints/sha/resnet50/stage2/best_mae_0.pth --dataset sha --model clip_resnet50 --out_png sha_resnet50.png --names "004.jpg,043.jpg,086.jpg"

python visualize_stage2.py --config checkpoints/sha/vit_b_16/stage2_size224_56/config.yaml --checkpoint checkpoints/sha/vit_b_16/stage2_size224_56/best_mae_0.pth --dataset sha --model clip_vit_b_16 --out_png sha_vit_b_16.png --indices "10,93,153"

#python visualize_stage2.py --config checkpoints/shb/vit_b_16/stage2/config_stage2.yaml --checkpoint checkpoints/shb/vit_b_16/stage2/best_mae_0.pth --dataset shb --model clip_vit_b_16 --out_png shb_vit_b_16.png


#srun python visualize_stage2.py --config checkpoints/shb/resnet50/stage2/config_stage2.yaml --checkpoint checkpoints/shb/resnet50/stage2/best_mae_0.pth --dataset shb --model clip_resnet50 --out_png shb_resnet50.png
#python visualize_stage2.py --config checkpoints/qnrf/vit_b_16/stage2/config_stage2.yaml --checkpoint checkpoints/qnrf/vit_b_16/stage2/best_mae_0.pth --dataset qnrf --model clip_vit_b_16 --out_png qnrf_vit_b_16.png
srun python visualize_stage2.py   --config checkpoints/qnrf/resnet50/stage2_inputsize448/training_config.json   --checkpoint checkpoints/qnrf/resnet50/stage2_inputsize448/best_mae_0.pth   --dataset qnrf   --model clip_resnet50   --out_png qnrf_resnet50.png   --indices "20,44,55"


#STAGE3

#python evaluation_stage3.py --config_zip checkpoints/shb/vit_b_16/stage1_v2/config_stage1.yaml --config_clip checkpoints/shb/vit_b_16/stage2_v2/config.yaml --ckpt_zip checkpoints/shb/vit_b_16/stage1_v2/best_model.pth --ckpt_clip checkpoints/shb/vit_b_16/stage2_v2/best_mae_0.pth --dataset_root shb 



