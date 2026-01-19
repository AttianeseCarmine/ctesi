import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

from argparse import ArgumentParser
import os, json
import yaml 
import sys  

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import standardize_dataset_name
from models import get_model

from utils import setup, cleanup, init_seeds, get_logger, get_config, barrier
from utils import get_dataloader, get_loss_fn, get_optimizer, load_checkpoint, save_checkpoint
from utils import get_writer, update_train_result, update_eval_result, log
from train_stage2 import train
from eval import evaluate

parser = ArgumentParser(description="Train an EBC model (Stage 2).")

# --- ARGOMENTO CONFIG (Corretto Default) ---
# Se l'utente non specifica nulla, prova a caricare configs/config.yaml
parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the .yaml configuration file.")

# Parameters for model
parser.add_argument("--model", type=str, default="clip_vit_b_16", help="The model to train.")
parser.add_argument("--input_size", type=int, default=448, help="The size of the input image.")
parser.add_argument("--reduction", type=int, default=16, choices=[8, 16, 32], help="Reduction factor.")
parser.add_argument("--regression", action="store_true", help="Use blockwise regression.")
parser.add_argument("--truncation", type=int, default=4, help="The truncation of the count.")
parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"])
parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"])
parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"])
parser.add_argument("--num_vpt", type=int, default=32)
parser.add_argument("--vpt_drop", type=float, default=0.0)
parser.add_argument("--shallow_vpt", action="store_true")

# Parameters for dataset
parser.add_argument("--dataset", type=str, required=False)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_crops", type=int, default=1)
parser.add_argument("--min_scale", type=float, default=1.0)
parser.add_argument("--max_scale", type=float, default=2.0)
parser.add_argument("--brightness", type=float, default=0.1)
parser.add_argument("--contrast", type=float, default=0.1)
parser.add_argument("--saturation", type=float, default=0.1)
parser.add_argument("--hue", type=float, default=0.0)
parser.add_argument("--kernel_size", type=int, default=5)
parser.add_argument("--saltiness", type=float, default=1e-3)
parser.add_argument("--spiciness", type=float, default=1e-3)
parser.add_argument("--jitter_prob", type=float, default=0.2)
parser.add_argument("--blur_prob", type=float, default=0.2)
parser.add_argument("--noise_prob", type=float, default=0.5)

# Parameters for evaluation
parser.add_argument("--sliding_window", action="store_true")
parser.add_argument("--stride", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--resize_to_multiple", action="store_true")
parser.add_argument("--zero_pad_to_multiple", action="store_true")

# Loss & Optimizer
parser.add_argument("--weight_count_loss", type=float, default=1.0)
parser.add_argument("--count_loss", type=str, default="dmcount", choices=["mae", "mse", "dmcount"])
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)

# Scheduler & Train
parser.add_argument("--warmup_epochs", type=int, default=50)
parser.add_argument("--warmup_lr", type=float, default=1e-6)
parser.add_argument("--T_0", type=int, default=5)
parser.add_argument("--T_mult", type=int, default=2)
parser.add_argument("--eta_min", type=float, default=1e-7)
parser.add_argument("--total_epochs", type=int, default=200)
parser.add_argument("--eval_start", type=int, default=10)
parser.add_argument("--eval_freq", type=int, default=1)
parser.add_argument("--save_freq", type=int, default=5)
parser.add_argument("--save_best_k", type=int, default=3)
parser.add_argument("--amp", action="store_true")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)


def run(local_rank: int, nprocs: int, args: ArgumentParser) -> None:
    print(f"Rank {local_rank} process among {nprocs} processes.")
    init_seeds(args.seed + local_rank)
    setup(local_rank, nprocs)
    
    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:0"
    ddp = nprocs > 1

    # Caricamento Bins (Logica originale)
    if args.regression:
        bins, anchor_points = None, None
    else:
        # Assicurati che 'configs/reduction_16.json' esista se usi reduction 16
        with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
            config = json.load(f)[str(args.truncation)][args.dataset]
        bins = config["bins"][args.granularity]
        anchor_points = config["anchor_points"][args.granularity]["average"] if args.anchor_points == "average" else config["anchor_points"][args.granularity]["middle"]
        bins = [(float(b[0]), float(b[1])) for b in bins]
        anchor_points = [float(p) for p in anchor_points]

    args.bins = bins
    args.anchor_points = anchor_points

    # Init Model
    model = get_model(
        backbone=args.model,
        input_size=args.input_size, 
        reduction=args.reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=not args.shallow_vpt
    ).to(device)

    grad_scaler = GradScaler() if args.amp else None
    loss_fn = get_loss_fn(args).to(device)
    optimizer, scheduler = get_optimizer(args, model)

    # --- DIRECTORIES (Nuova Logica Ordinata: .../stage2) ---
    config_name = f"{args.model}_{args.dataset}"
    args.ckpt_dir = os.path.join(current_dir, "checkpoints", args.dataset, config_name, "stage2")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # *** NOVITÀ: Salvataggio Configurazione su File ***
    if local_rank == 0:
        config_save_path = os.path.join(args.ckpt_dir, "config_stage2.yaml")
        try:
            with open(config_save_path, 'w') as f:
                # Salviamo vars(args) che contiene tutti i parametri usati
                yaml.dump(vars(args), f, default_flow_style=False)
            print(f"📄 Configuration saved to {config_save_path}")
        except Exception as e:
            print(f"⚠️ Could not save config yaml: {e}")
    # -------------------------------------------------------

    model, optimizer, scheduler, grad_scaler, start_epoch, loss_info, hist_val_scores, best_val_scores = load_checkpoint(args, model, optimizer, scheduler, grad_scaler)

    if local_rank == 0:
        model_without_ddp = model
        writer = get_writer(args.ckpt_dir)
        logger = get_logger(os.path.join(args.ckpt_dir, "train_stage2.log"))
        logger.info(get_config(vars(args), mute=False))
        val_loader = get_dataloader(args, split="val", ddp=False)

    args.batch_size = int(args.batch_size / nprocs)
    args.num_workers = int(args.num_workers / nprocs)
    train_loader, sampler = get_dataloader(args, split="train", ddp=ddp)

    model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank], output_device=local_rank) if ddp else model

    for epoch in range(start_epoch, args.total_epochs + 1):
        if local_rank == 0:
            log(logger, epoch, args.total_epochs, message=f"\tlr: {optimizer.param_groups[0]['lr']:.3e}")

        if sampler is not None: sampler.set_epoch(epoch)

        model, optimizer, grad_scaler, loss_info = train(model, train_loader, loss_fn, optimizer, grad_scaler, device, local_rank, nprocs)
        scheduler.step()
        barrier(ddp)

        if local_rank == 0:
            eval = (epoch >= args.eval_start) and ((epoch - args.eval_start) % args.eval_freq == 0)
            update_train_result(epoch, loss_info, writer)
            log(logger, None, None, loss_info=loss_info, message="\n" * 2 if not eval else None)

            if eval:
                print("Evaluating Stage 2...")
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                model_without_ddp.load_state_dict(state_dict)
                curr_val_scores = evaluate(
                    model_without_ddp,
                    val_loader,
                    device,
                    args.sliding_window,
                    args.window_size,
                    args.stride,
                )
                hist_val_scores, best_val_scores = update_eval_result(epoch, curr_val_scores, hist_val_scores, best_val_scores, writer, state_dict, os.path.join(args.ckpt_dir))
                log(logger, None, None, None, curr_val_scores, best_val_scores, message="\n" * 3)
    
            if (epoch % args.save_freq == 0):
                save_checkpoint(epoch + 1, (model.module if ddp else model).state_dict(), optimizer.state_dict(),
                    scheduler.state_dict(), grad_scaler.state_dict() if grad_scaler else None,
                    loss_info, hist_val_scores, best_val_scores, args.ckpt_dir)

        barrier(ddp)

    if local_rank == 0:
        writer.close()
        cleanup(ddp)


def load_config_and_update_args(args):
    # Se args.config è None, usa il default o esci (grazie al default="configs/config.yaml" nel parser, questo non dovrebbe più essere None)
    if not args.config: 
        print("⚠️ No config file specified. Using defaults/cli args only.")
        return args

    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}...")
        with open(args.config, 'r') as f: cfg = yaml.safe_load(f)
    else:
        print(f"⚠️ Config file {args.config} not found! Continuing with CLI args.")
        return args

    def is_passed_in_cli(arg_name): return f"--{arg_name}" in sys.argv

    mapping = {
        "dataset": ["DATASET"],
        "model": ["MODEL"],             
        "input_size": ["INPUT_SIZE"],   
        "reduction": ["REDUCTION"],
        "truncation": ["TRUNCATION"],   # <--- Questo mancava e causava KeyError
        
        # Train Stage 2 Params
        "batch_size": ["TRAIN_STAGE2", "BATCH_SIZE"],
        "num_workers": ["TRAIN_STAGE2", "NUM_WORKERS"],
        "lr": ["TRAIN_STAGE2", "LR_HEAD"],
        "weight_decay": ["TRAIN_STAGE2", "WEIGHT_DECAY"],
        "total_epochs": ["TRAIN_STAGE2", "TOTAL_EPOCHS"],
        "eval_freq": ["TRAIN_STAGE2", "EVAL_FREQ"],
        "warmup_epochs": ["TRAIN_STAGE2", "WARMUP_EPOCHS"],
        
        "weight_count_loss": ["LOSS_STAGE2", "WEIGHT_COUNT_LOSS"],
        "count_loss": ["LOSS_STAGE2", "COUNT_LOSS"],
        "amp": ["TRAIN_STAGE2", "AMP"]
    }

    updated_count = 0
    for arg_key, keys_path in mapping.items():
        if not is_passed_in_cli(arg_key):
            try:
                val = cfg
                for k in keys_path: val = val[k]
                setattr(args, arg_key, val)
                updated_count += 1
            except (KeyError, TypeError): pass
    
    print(f"Stage 2 Config Loaded: Updated {updated_count} args.")
    return args


def main():
    args = parser.parse_args()
    args = load_config_and_update_args(args)
    
    if args.dataset is None:
        raise ValueError("Dataset name must be provided.")
    args.dataset = standardize_dataset_name(args.dataset)

    # ... logica sliding window originale ...
    if args.sliding_window:
        args.window_size = args.input_size if args.window_size is None else args.window_size
        args.stride = args.input_size if args.stride is None else args.stride
        assert not (args.zero_pad_to_multiple and args.resize_to_multiple), "Cannot use both."
    else:
        args.window_size = None; args.stride = None

    args.nprocs = torch.cuda.device_count()
    print(f"Using {args.nprocs} GPUs.")
    if args.nprocs > 1:
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args))
    else:
        run(0, 1, args)

if __name__ == "__main__":
    main()