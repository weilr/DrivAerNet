# train.py
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu

Training script for RegDGCNN pressure field prediction model on the DrivAerNet++ dataset.
This version includes distributed training support for multi-GPU acceleration.
"""
import argparse
import datetime
import logging
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DeepSurrogates.utils import init_logger, EarlyStopping, progress
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from model_pressure import RegDGCNN_pressure
from utils import setup_seed

if platform.system() == "Windows":
    proj_path = os.path.dirname(os.getcwd())
else:
    proj_path = os.getcwd()
os.chdir(os.getcwd())

writer = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train pressure prediction models on DrivAerNet++')

    # Basic settings
    parser.add_argument('--exp_name', type=str, default='PressurePrediction', help='Experiment name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # Data settings
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--subset_dir', type=str, required=True, help='Path to train/val/test splits')
    parser.add_argument('--cache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--model_path', type=str, help='Path to test model directory')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points to sample')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--test_only', action='store_true', help='Only test the model, no training')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPUs to use (comma-separated)')

    # Model settings
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--k', type=int, default=40, help='Number of nearest neighbors')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of output channels')

    return parser.parse_args()


def initialize_model(args, local_rank):
    """Initialize and return the RegDGCNN model."""
    args = vars(args)

    model = RegDGCNN_pressure(args).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True, output_device=local_rank
    )
    return model


def train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for data, targets in progress(train_dataloader, desc=f"Epoch {epoch + 1} [Training]"):
        data, targets = data.squeeze(1).to(local_rank), targets.squeeze(1).to(local_rank)
        targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(1), targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)


# def validate(model, val_dataloader, criterion, local_rank, epoch):
#     """Validate the model and measure all_reduce overhead."""
#     model.eval()
#     total_loss = 0
#     num_batches = len(val_dataloader)
#
#     with torch.no_grad():
#         for data, targets in progress(val_dataloader, desc=f"Epoch {epoch + 1} [Validation]"):
#             data, targets = data.squeeze(1).to(local_rank), targets.squeeze(1).to(local_rank)
#             targets = (targets - PRESSURE_MEAN) / PRESSURE_STD
#
#             outputs = model(data)
#             loss = criterion(outputs.squeeze(1), targets)
#             total_loss += loss.item()
#
#         total_loss_tensor = torch.tensor(total_loss / num_batches).to(local_rank)
#         dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
#         avg_loss = total_loss_tensor.item() / dist.get_world_size()
#
#         return avg_loss

def validate(model, val_dataloader, criterion, local_rank, epoch):
    """Validate the model and measure all_reduce overhead."""
    model.eval()
    total_loss = 0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for data, targets in progress(val_dataloader, desc=f"Epoch {epoch + 1} [Validation]"):
            data, targets = data.squeeze(1).to(local_rank), targets.squeeze(1).to(local_rank)
            targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

            outputs = model(data)
            loss = criterion(outputs.squeeze(1), targets)
            total_loss += loss.item()

    # 测量 all_reduce 开销
    start_time = time.time()
    total_loss_tensor = torch.tensor(total_loss / num_batches).to(local_rank)
    handle = dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM, async_op=True)
    handle.wait()
    all_reduce_time = time.time() - start_time
    logging.info(f"[Rank {local_rank}] Epoch {epoch + 1} all_reduce time: {all_reduce_time:.6f}s")

    avg_loss = total_loss_tensor.item() / dist.get_world_size()
    return avg_loss, all_reduce_time  # 返回损失和通信开销

def test_model(model, test_dataloader, criterion, local_rank, exp_dir):
    """Test the model and calculate metrics."""
    model.eval()
    total_mse, total_mae = 0, 0
    total_rel_l2, total_rel_l1 = 0, 0
    total_inference_time = 0
    total_samples = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data, targets in tqdm(test_dataloader, desc="[Testing]"):
            start_time = time.time()

            data, targets = data.squeeze(1).to(local_rank), targets.squeeze(1).to(local_rank)
            normalized_targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

            outputs = model(data)
            normalized_outputs = outputs.squeeze(1)

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # Calculate metrics
            mse = criterion(normalized_outputs, normalized_targets)
            mae = F.l1_loss(normalized_outputs, normalized_targets)

            # Calculate relative errors
            rel_l2 = torch.mean(torch.norm(normalized_outputs - normalized_targets, p=2, dim=-1) /
                                torch.norm(normalized_targets, p=2, dim=-1))
            rel_l1 = torch.mean(torch.norm(normalized_outputs - normalized_targets, p=1, dim=-1) /
                                torch.norm(normalized_targets, p=1, dim=-1))

            batch_size = targets.size(0)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size
            total_rel_l2 += rel_l2.item() * batch_size
            total_rel_l1 += rel_l1.item() * batch_size
            total_samples += batch_size

            # Store normalized predictions and targets for R² calculation
            all_outputs.append(normalized_outputs.cpu())
            all_targets.append(normalized_targets.cpu())

    # Aggregate results across all processes
    total_mse_tensor = torch.tensor(total_mse).to(local_rank)
    total_mae_tensor = torch.tensor(total_mae).to(local_rank)
    total_rel_l2_tensor = torch.tensor(total_rel_l2).to(local_rank)
    total_rel_l1_tensor = torch.tensor(total_rel_l1).to(local_rank)
    total_samples_tensor = torch.tensor(total_samples).to(local_rank)

    dist.reduce(total_mse_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_mae_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_rel_l2_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_rel_l1_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_samples_tensor, dst=0, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        # Calculate aggregated metrics
        avg_mse = total_mse_tensor.item() / total_samples_tensor.item()
        avg_mae = total_mae_tensor.item() / total_samples_tensor.item()
        avg_rel_l2 = total_rel_l2_tensor.item() / total_samples_tensor.item()
        avg_rel_l1 = total_rel_l1_tensor.item() / total_samples_tensor.item()

        # Calculate R² score - only on rank 0 with locally collected data
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        ss_res = np.sum((all_targets - all_outputs) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate max MAE
        max_mae = np.max(np.abs(all_targets - all_outputs))

        # print(f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max MAE: {max_mae:.6f}, Test R²: {r_squared:.4f}")
        # print(f"Relative L2 Error: {avg_rel_l2:.6f}, Relative L1 Error: {avg_rel_l1:.6f}")
        # print(f"Total inference time: {total_inference_time:.2f}s for {total_samples_tensor.item()} samples")

        logging.info(
            f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max MAE: {max_mae:.6f}, Test R²: {r_squared:.4f}")
        logging.info(f"Relative L2 Error: {avg_rel_l2:.6f}, Relative L1 Error: {avg_rel_l1:.6f}")
        logging.info(f"Total inference time: {total_inference_time:.2f}s for {total_samples_tensor.item()} samples")
        # Save metrics to a text file
        metrics_file = os.path.join(exp_dir, 'test_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Test MSE: {avg_mse:.6f}\n")
            f.write(f"Test MAE: {avg_mae:.6f}\n")
            f.write(f"Max MAE: {max_mae:.6f}\n")
            f.write(f"R² Score: {r_squared:.6f}\n")
            f.write(f"Relative L2 Error: {avg_rel_l2:.6f}\n")
            f.write(f"Relative L1 Error: {avg_rel_l1:.6f}\n")
            f.write(f"Total inference time: {total_inference_time:.2f}s for {total_samples_tensor.item()} samples\n")


def train_and_evaluate(rank, world_size, args):
    """Main function for distributed training and evaluation with communication overhead tracking."""
    setup_seed(args.seed)

    # Initialize process group for DDP  gloo/nccl
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    local_rank = rank
    torch.cuda.set_device(local_rank)
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(proj_path, 'logs/PressurePred', f'{args.exp_name}', f'[{rank}]training.log')),
            logging.StreamHandler()
        ]
    )
    # Set up logging (only on rank 0)
    if local_rank == 0:
        global writer
        logging.info(f"Starting training with {world_size} GPUs")
        sys.stdout.reconfigure(line_buffering=True)
        init_logger(log_file=os.path.join(proj_path, 'logs/PressurePred', f'{args.exp_name}'), log_name=f'training.log')
        logging.info(f"[Main] Initializing at the {proj_path} path in the {platform.system()} system.")
        if writer is None:
            logdir = os.path.join(proj_path, 'runs', f'{args.exp_name}')
            logging.info(f"[Main] Initializing TensorBoard at {logdir}")
            writer = SummaryWriter(logdir)  # tensorboard --logdir runs

    # Initialize model
    model = initialize_model(args, local_rank)

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params}")

    # Prepare DataLoaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        args.dataset_path, args.subset_dir, args.num_points,
        args.batch_size, world_size, rank, args.cache_dir,
        args.num_workers
    )

    # Log dataset info
    if local_rank == 0:
        logging.info(
            f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches")

    training_start_time = time.time()

    # Set up criterion, optimizer, and scheduler
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True, eps=float('1e-15'))
    early_stopping = EarlyStopping(patience=50, verbose=True)

    best_model_path = os.path.join(proj_path, 'logs/PressurePred', f'{args.exp_name}',
                                   f'{args.exp_name}_best_model.pth')
    final_model_path = os.path.join(proj_path, 'logs/PressurePred', f'{args.exp_name}',
                                    f'{args.exp_name}_final_model.pth')

    # Check if test_only and model exists
    if args.test_only and os.path.exists(args.model_path):
        if local_rank == 0:
            logging.info("Loading best model for testing only")
            print("Testing the best model:")
        model.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{local_rank}'))
        test_model(model, test_dataloader, criterion, local_rank, os.path.join('experiments', args.exp_name))
        dist.destroy_process_group()
        return

    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    total_all_reduce_time = 0.0
    total_broadcast_time = 0.0

    if local_rank == 0:
        logging.info(f"Starting training for {args.epochs} epochs")

    # Training loop
    for epoch in range(args.epochs):
        # Set epoch for the DistributedSampler
        train_dataloader.sampler.set_epoch(epoch)

        # Training
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank, epoch)

        # Validation
        val_loss, all_reduce_time = validate(model, val_dataloader, criterion, local_rank, epoch)
        total_all_reduce_time += all_reduce_time

        # Record losses
        if local_rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            logging.info(f"[Epoch {epoch + 1}/{args.epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            writer.add_scalars("RegDGCNN_Loss", {'train': train_loss, 'test': val_loss}, epoch + 1)

            scheduler.step(val_loss)
            early_stopping(val_loss, model, best_model_path)
        else:
            scheduler.step(val_loss)

        # 测量 broadcast 开销
        start_time = time.time()
        early_stop_tensor = torch.tensor(early_stopping.early_stop if local_rank == 0 else False, dtype=torch.bool).to(local_rank)
        dist.broadcast(early_stop_tensor, src=0)
        broadcast_time = time.time() - start_time
        total_broadcast_time += broadcast_time
        logging.info(f"[Rank {local_rank}] Epoch {epoch + 1} broadcast time: {broadcast_time:.6f}s")

        early_stop = early_stop_tensor.item()
        if early_stop:
            logging.info(f"[Rank {local_rank}][Epoch {epoch + 1}] Early stopping triggered, exiting training loop")
            if local_rank == 0:
                logging.info(f"Early stopping triggered in epoch {epoch + 1}")
            break

    # Log total communication overhead
    logging.info(f"[Rank {local_rank}] Total all_reduce time: {total_all_reduce_time:.6f}s")
    logging.info(f"[Rank {local_rank}] Total broadcast time: {total_broadcast_time:.6f}s")

    # Aggregate communication overhead across processes (optional)
    all_reduce_time_tensor = torch.tensor(total_all_reduce_time).to(local_rank)
    broadcast_time_tensor = torch.tensor(total_broadcast_time).to(local_rank)
    dist.reduce(all_reduce_time_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(broadcast_time_tensor, dst=0, op=dist.ReduceOp.SUM)
    if local_rank == 0:
        avg_all_reduce_time = all_reduce_time_tensor.item() / world_size
        avg_broadcast_time = broadcast_time_tensor.item() / world_size
        logging.info(f"Average all_reduce time across {world_size} processes: {avg_all_reduce_time:.6f}s")
        logging.info(f"Average broadcast time across {world_size} processes: {avg_broadcast_time:.6f}s")

    # Save final model
    if local_rank == 0:
        training_duration = time.time() - training_start_time
        logging.info(f"Total training time: {training_duration:.2f}s")
        logging.info(f"Communication overhead ratio: {(total_all_reduce_time + total_broadcast_time) / training_duration * 100:.2f}%")
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

    # Make sure all processes sync up before testing
    logging.info(f"[Rank {rank}] Reaching dist.barrier() ...")
    dist.barrier()
    logging.info(f"[Rank {rank}] Passed dist.barrier()")

    test_metrics_path = os.path.join(proj_path, 'logs/PressurePred', f'{args.exp_name}')
    if local_rank == 0:
        logging.info("Testing the final model")
    test_model(model, test_dataloader, criterion, local_rank, test_metrics_path)

    if local_rank == 0:
        logging.info("Testing the best model")
    model.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{local_rank}'))
    test_model(model, test_dataloader, criterion, local_rank, test_metrics_path)

    dist.destroy_process_group()


def init(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.exp_name = f"{args.exp_name}_{timestamp}_{args.num_points}nP_{args.k}k_{args.emb_dims}emb_{args.dropout}dp"

    logging.info("[Config] Training configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg:<20}: {value}")


def main():
    """Main function to parse arguments and start training."""
    args = parse_args()
    init(args)

    # Set the master address and port for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Set visible GPUs
    gpu_list = args.gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # Count number of GPUs to use
    world_size = len(gpu_list.split(','))

    # Create experiment directory
    exp_dir = os.path.join(proj_path, 'logs/PressurePred', f'{args.exp_name}')
    os.makedirs(exp_dir, exist_ok=True)

    # Start distributed training
    mp.spawn(train_and_evaluate, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
