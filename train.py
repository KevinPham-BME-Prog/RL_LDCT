import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import random
import logging
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from torch.amp import autocast
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import NAS_SR_Model
from trainer import DQNTrainer
from dataset import ImageDataset, get_transforms, get_validation_transform, custom_collate_fn
from utils import validate, plot_metrics, calculate_block_flops, calculate_block_parameters, calculate_ssim


def register_new_skip_params(model, optimizer):
    """Add lazily created skip combiner parameters to the optimizer."""
    if hasattr(model, "_new_skip_params") and model._new_skip_params:
        for params in model._new_skip_params:
            optimizer.add_param_group({"params": params})
        model._new_skip_params = []





def main(args):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("log.txt"), logging.StreamHandler()])
    logger = logging.getLogger()

    # --- Device and Seed Setup ---
    train_device = torch.device(f"cuda:{args.train_gpu}" if torch.cuda.is_available() else "cpu")
    val_device = train_device
    args.device = train_device
    logger.info(f"Using device: {train_device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    # --- Data Loading ---
    logger.info("Setting up datasets...")
    train_transform = get_transforms(args.lr_patch_size)
    train_dataset = ImageDataset(root_dir=args.div2k_train_hr_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_transform = get_transforms(args.lr_patch_size)  # use val transform, match patch size
    val_dataset = ImageDataset(root_dir=args.div2k_valid_hr_dir, transform=train_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=custom_collate_fn)
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- Model and Trainer Initialization ---
    logger.info("Initializing model and trainer...")
    model = NAS_SR_Model(args).to(train_device)
    
    block_flops = calculate_block_flops(model, args.lr_patch_size, args.num_features, train_device)
    block_params = calculate_block_parameters(model, train_device)
    trainer = DQNTrainer(model, block_flops, block_params, args)
    
    # Clear replay buffer to remove any inconsistent states from previous runs
    trainer.replay_buffer.memory.clear()
    trainer.replay_buffer.position = 0
    logger.info("Replay buffer cleared to ensure consistent state shapes")

    # --- Schedulers and Scaler ---
    scaler = GradScaler() 
    worker_scheduler = ReduceLROnPlateau(trainer.worker_optimizer, mode='max', factor=0.7, patience=1)
    # Controller scheduler is now built into DQNTrainer (CosineAnnealingWarmRestarts)

    # --- Training Loop ---
    best_psnr = 0.0
    best_architecture = []
    train_mse_history, train_reward_history, val_psnr_history = [], [], []

    for epoch in range(args.num_epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for x_lr_batch, y_hr_batch in progress_bar:
            x_lr_batch, y_hr_batch = x_lr_batch.to(train_device), y_hr_batch.to(train_device)

            # Process each sample in the batch individually
            for i in range(x_lr_batch.size(0)):
                x_lr = x_lr_batch[i].unsqueeze(0)  # [1, 1, H, W]
                y_hr = y_hr_batch[i].unsqueeze(0)  # [1, 1, H, W]

                current_architecture = []
                features = model.initial_conv(x_lr)
                
                states = []
                actions = []
                next_states = []

                # Build architecture layer by layer WITH U-Net downsampling/upsampling and skip connections
                feat_mat = []  # Store features for skip connections
                
                for i in range(args.num_layers):
                    pooled_state = model.state_pooler(features)
                    state = model.project_state(pooled_state)
                    state = state.detach()
                    if state.size(-1) < args.num_features:
                        state = torch.nn.functional.pad(state, (0, args.num_features - state.size(-1)))

                    action = trainer.select_action(state)
                    current_architecture.append(action.item())

                    states.append(state.view(-1).cpu())
                    actions.append(action.cpu())

                    layer_idx = action.item()
                    # pick per-layer UNet block
                    chosen_layer = (model.unet_layer_options[layer_idx][i]
                                    if model.UNet_initialization == 1
                                    else model.layer_options[layer_idx])

                    if model.UNet_initialization == 1:
                        if i < args.num_layers // 2:
                            if not model.input_skip_only:
                                feat_mat.append(features.clone().detach())
                            features = chosen_layer(features)
                            if i < len(model.downsamplers):
                                features = model.downsamplers[i](features)
                        elif i == args.num_layers // 2:
                            features = chosen_layer(features)
                        else:
                            upsample_idx = i - (args.num_layers // 2 + 1)
                            skip_connection = None
                            if not model.input_skip_only and feat_mat:
                                skip_connection = feat_mat.pop()
                            if 0 <= upsample_idx < len(model.upsamplers):
                                features = model.upsamplers[upsample_idx](features)

                            features = chosen_layer(features)

                            if skip_connection is not None:
                                if skip_connection.shape[-2:] != features.shape[-2:]:
                                    raise ValueError(
                                        f"Spatial mismatch for skip connection at layer {i}: "
                                        f"decoder {features.shape[-2:]}, skip {skip_connection.shape[-2:]}"
                                    )
                                if len(model.skip_combiners) == 0:
                                    raise ValueError(
                                        "Skip combiners are not initialized but a skip connection is available."
                                    )
                                comb_idx = min(upsample_idx, len(model.skip_combiners) - 1)
                                features = model.fuse_local_skip(features, skip_connection, comb_idx)
                    else:
                        features = features + chosen_layer(features)


                    pooled_next = model.state_pooler(features)
                    next_state = model.project_state(pooled_next)
                    next_state = next_state.detach()
                    if next_state.size(-1) < args.num_features:
                        next_state = torch.nn.functional.pad(next_state, (0, args.num_features - next_state.size(-1)))
                    next_states.append(next_state.view(-1).cpu())

                with torch.no_grad():
                    # Manual forward matching training path
                    tmp_feat = model.initial_conv(x_lr)
                    tmp_initial = tmp_feat
                    tmp_feat_mat = []
                    
                    for j in range(args.num_layers):
                        act = current_architecture[j]
                        block = (model.unet_layer_options[act][j]
                                 if model.UNet_initialization == 1
                                 else model.layer_options[act])

                        if model.UNet_initialization == 1:
                            if j < args.num_layers // 2:
                                if not model.input_skip_only:
                                    tmp_feat_mat.append(tmp_feat.clone().detach())
                                tmp_feat = block(tmp_feat)
                                if j < len(model.downsamplers):
                                    tmp_feat = model.downsamplers[j](tmp_feat)
                            elif j == args.num_layers // 2:
                                tmp_feat = block(tmp_feat)
                            else:
                                up_idx = j - (args.num_layers // 2 + 1)
                                skip = None
                                if not model.input_skip_only and tmp_feat_mat:
                                    skip = tmp_feat_mat.pop()
                                if 0 <= up_idx < len(model.upsamplers):
                                    tmp_feat = model.upsamplers[up_idx](tmp_feat)
                                tmp_feat = block(tmp_feat)
                                if skip is not None:
                                    if skip.shape[-2:] != tmp_feat.shape[-2:]:
                                        raise ValueError(
                                            f"Spatial mismatch for skip connection at layer {j} during reward forward: "
                                            f"decoder {tmp_feat.shape[-2:]}, skip {skip.shape[-2:]}"
                                        )
                                    if len(model.skip_combiners) == 0:
                                        raise ValueError(
                                            "Skip combiners are not initialized but a skip connection is available."
                                        )
                                    comb_idx = min(up_idx, len(model.skip_combiners) - 1)
                                    tmp_feat = model.fuse_local_skip(tmp_feat, skip, comb_idx)
                        else:
                            # Standard residual path
                            tmp_feat = tmp_feat + block(tmp_feat)
                    # Fuse global skips before final reconstruction (both paths)
                    tmp_feat = model.combine_with_global_skips(tmp_feat, x_lr, tmp_initial)
                    y_pred_final = model.upsampler(tmp_feat)
                    if y_pred_final.shape != x_lr.shape:
                        raise RuntimeError(
                            f"Manual forward output shape mismatch: pred {y_pred_final.shape}, input {x_lr.shape}"
                        )
                    y_pred_final = model.output_activation(y_pred_final)

                # Sanity check: shapes must match
                if y_pred_final.shape != y_hr.shape:
                    raise RuntimeError(f"Shape mismatch: y_pred_final={y_pred_final.shape}, y_hr={y_hr.shape}.")
                
                mse_reward = -model.mse_loss_fn(y_pred_final, y_hr).item()
                ssim_reward = calculate_ssim(y_pred_final, y_hr).item()
                reward = (args.ssim_weight * ssim_reward) + ((1 - args.ssim_weight) * mse_reward)
                
                # Store experiences in replay buffer (on CPU to save GPU memory)
                for layer_idx in range(args.num_layers):
                    trainer.replay_buffer.push(
                        states[layer_idx],          # Already on CPU
                        actions[layer_idx],         # Already on CPU
                        next_states[layer_idx],     # Already on CPU
                        torch.tensor([reward])      # CPU tensor
                    )
                
                # Optimize controller
                trainer.optimize_controller()
                
                # Free memory
                del y_pred_final, features, feat_mat
                torch.cuda.empty_cache()
        
            # Ensure optimizer tracks any newly created skip projection params
            register_new_skip_params(model, trainer.worker_optimizer)

            # Train worker network on the last architecture of the batch
            trainer.worker_optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda'):
                feat = model.initial_conv(x_lr_batch)  # keep 64x64
                initial_batch_feat = feat
                feat_mat = []
                
                for j in range(args.num_layers):
                    act = current_architecture[j]
                    block = (model.unet_layer_options[act][j]
                             if model.UNet_initialization == 1
                             else model.layer_options[act])

                    if model.UNet_initialization == 1:
                        if j < args.num_layers // 2:
                            if not model.input_skip_only:
                                feat_mat.append(feat.clone())
                            feat = block(feat)
                            if j < len(model.downsamplers):
                                feat = model.downsamplers[j](feat)
                        elif j == args.num_layers // 2:
                            feat = block(feat)
                        else:
                            up_idx = j - (args.num_layers // 2 + 1)
                            skip = None
                            if not model.input_skip_only and feat_mat:
                                skip = feat_mat.pop()
                            if 0 <= up_idx < len(model.upsamplers):
                                feat = model.upsamplers[up_idx](feat)
                            feat = block(feat)
                            if skip is not None:
                                if skip.shape[-2:] != feat.shape[-2:]:
                                    raise ValueError(
                                        f"Spatial mismatch for skip connection at layer {j} during worker forward: "
                                        f"decoder {feat.shape[-2:]}, skip {skip.shape[-2:]}"
                                    )
                                if len(model.skip_combiners) == 0:
                                    raise ValueError(
                                        "Skip combiners are not initialized but a skip connection is available."
                                    )
                                comb_idx = min(up_idx, len(model.skip_combiners) - 1)
                                feat = model.fuse_local_skip(feat, skip, comb_idx)
                    else:
                        feat = feat + block(feat)
                # Fuse global skips before final reconstruction (both paths)
                feat = model.combine_with_global_skips(feat, x_lr_batch, initial_batch_feat)
                y_pred = model.upsampler(feat)
                if y_pred.shape != x_lr_batch.shape:
                    raise RuntimeError(
                        f"Worker forward output shape mismatch: pred {y_pred.shape}, input {x_lr_batch.shape}"
                    )
                y_pred = model.output_activation(y_pred)

                if y_pred.shape != y_hr_batch.shape:
                    raise RuntimeError(f"Shape mismatch during worker training: y_pred={y_pred.shape}, y_hr_batch={y_hr_batch.shape}.")
                worker_loss = model.mse_loss_fn(y_pred, y_hr_batch)

            scaler.scale(worker_loss).backward()
            scaler.step(trainer.worker_optimizer)
            scaler.update()
            
            train_mse_history.append(worker_loss.item())
            train_reward_history.append(reward)
            arch_indices = torch.tensor(current_architecture, device=train_device, dtype=torch.long)
            total_params = block_params[arch_indices].sum().item()
            total_flops = block_flops[arch_indices].sum().item()
            
            progress_bar.set_postfix(
                MSE=f'{worker_loss.item():.4f}', 
                Reward=f'{reward:.4f}',
                Arch=f'{"-".join(map(str, current_architecture))}',
                Params_M=f'{total_params:.3f}',
                GFLOPs=f'{total_flops:.2f}'
            )
            
            # Free worker memory
            del y_pred, feat, feat_mat
            torch.cuda.empty_cache()
            
        if epoch % args.target_update == 0:
            trainer.update_target_network()
            logger.info(f"🔄️ Target network updated")

        if (epoch + 1) % args.validation_lag == 0:
            logger.info("Running validation...")
            val_arch = []

            with torch.no_grad():
                model.eval()
                features = model.initial_conv(torch.randn(1, 1, args.lr_patch_size, args.lr_patch_size).to(val_device))
                feat_mat = []
                
                for layer_idx in range(args.num_layers):
                    pooled_state = model.state_pooler(features)
                    state = model.project_state(pooled_state)
                    if state.size(-1) < args.num_features:
                        state = torch.nn.functional.pad(state, (0, args.num_features - state.size(-1)))
                    q_values, _ = model.controller(state)
                    action = q_values.max(1)[1].item()
                    val_arch.append(action)

                    block = (model.unet_layer_options[action][layer_idx]
                             if model.UNet_initialization == 1
                             else model.layer_options[action])

                    if model.UNet_initialization == 1:
                        if layer_idx < args.num_layers // 2:
                            if not model.input_skip_only:
                                feat_mat.append(features.clone().detach())  # Store BEFORE block execution
                            features = block(features)
                            if layer_idx < len(model.downsamplers):
                                features = model.downsamplers[layer_idx](features)
                        elif layer_idx == args.num_layers // 2:
                            features = block(features)
                        else:
                            up_idx = layer_idx - (args.num_layers // 2 + 1)
                            skip = None
                            if not model.input_skip_only and feat_mat:
                                skip = feat_mat.pop()
                            if 0 <= up_idx < len(model.upsamplers):
                                features = model.upsamplers[up_idx](features)
                            features = block(features)
                            if skip is not None:
                                if skip.shape[-2:] != features.shape[-2:]:
                                    raise ValueError(
                                        f"Spatial mismatch for validation skip at layer {layer_idx}: "
                                        f"decoder {features.shape[-2:]}, skip {skip.shape[-2:]}"
                                    )
                                if len(model.skip_combiners) == 0:
                                    raise ValueError(
                                        "Skip combiners are not initialized but a skip connection is available."
                                    )
                                comb_idx = min(up_idx, len(model.skip_combiners) - 1)
                                features = model.fuse_local_skip(features, skip, comb_idx)
                    else:
                        features = features + block(features)
            

            current_psnr = validate(model, val_loader, val_device, val_arch, epoch)
            val_psnr_history.append(current_psnr)
            
            worker_scheduler.step(current_psnr)
            trainer.step_controller_scheduler(epoch)
            logger.info(f"Controller LR: {trainer.get_controller_lr():.6f}")
    
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_architecture = val_arch
                logger.info(f"🎉 New best PSNR! {best_psnr:.4f} dB with arch {best_architecture}")
                torch.save({'model_state_dict': model.state_dict(), 'best_architecture': best_architecture}, 'best_nas_sr_model.pth')
        
        plot_metrics(train_mse_history, train_reward_history, val_psnr_history)

    logger.info(f"\n--- Training Complete ---\nBest PSNR: {best_psnr:.4f} dB\nBest Arch: {best_architecture}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NAS for SISR with DQN")
    
    # --- Paths and System ---
    parser.add_argument('--div2k_train_hr_dir', type=str, default=r"trainp1\train")
    parser.add_argument('--div2k_valid_hr_dir', type=str, default=r"trainp1\val")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--train_gpu', type=int, default=0)
    parser.add_argument('--validation_gpu', type=int, default=0)
    parser.add_argument('--seed',type=int , default=42)

    # --- Model Architecture ---
    parser.add_argument('--upscale_factor', type=int, default=4)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=7) #to make U-Net set up I needed to change this from 5 to 7
    parser.add_argument('--controller_hidden_dim', type=int, default=100)
    parser.add_argument('--window_size', type=int, default=48)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--use_UNet', type=int, default=0) #added this argument to switch to U-Net architecture
    parser.add_argument('--initial_UNet_input_dim', type=int, default=64) #added this argument to set the initial input dimension for U-Net
    parser.add_argument('--input_skip_only', type=int, default=0, choices=[0,1], help='Use only the low-dose input skip before reconstruction.')

    # --- Block Selection ---
    parser.add_argument('--use_resblock', type=int, default=1, choices=[0,1])
    parser.add_argument('--use_transformer', type=int, default=1, choices=[0,1])
    parser.add_argument('--use_mamba', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_gnn', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_lstmconv', type=int, default=1, choices=[0,1])

    # --- Training ---
    parser.add_argument('--lr_patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--worker_lr', type=float, default=1e-5) #2e-4 is the original 
    parser.add_argument('--controller_lr', type=float, default=2e-3) 
    parser.add_argument('--gradient_clipping', type=int, default=1, choices=[0,1])
    parser.add_argument('--validation_lag', type=int, default=25) #change it back to 10
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--ssim_weight', type=float, default=0.8, help='Weight for SSIM in the combined reward signal.')

    # --- DQN ---
    parser.add_argument('--replay_buffer_capacity', type=int, default=2500) # Smaller for non-stationarity
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=int, default=50000)
    parser.add_argument('--target_update', type=int, default=1)
    
    # --- ADD THESE MISSING ARGUMENTS FOR PER ---
    parser.add_argument('--per_alpha', type=float, default=0.6, help='Alpha for Prioritized Experience Replay')
    parser.add_argument('--per_beta_start', type=float, default=0.4, help='Initial beta for PER importance sampling')
    parser.add_argument('--per_beta_frames', type=int, default=100000, help='Number of frames over which to anneal beta to 1.0')

    # --- Controller LR Scheduler (CosineAnnealingWarmRestarts) ---
    parser.add_argument('--ctrl_scheduler_T0', type=int, default=50, help='Number of epochs for the first restart cycle')
    parser.add_argument('--ctrl_scheduler_Tmult', type=int, default=2, help='Multiplier for cycle length after each restart')
    parser.add_argument('--ctrl_scheduler_eta_min', type=float, default=1e-5, help='Minimum LR floor for controller scheduler')

    args = parser.parse_args()
    main(args)