import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from thop import profile
from torchvision import transforms
import numpy as np
import os
import logging # Added for logging
from torchmetrics.image import StructuralSimilarityIndexMeasure


def calculate_psnr(img1, img2):
    """Calculates the Peak Signal-to-Noise Ratio between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2, data_range=1.0):
    """Calculates the Structural Similarity Index Measure between two images."""
    # Note: data_range should be 1.0 for images with pixel values in [0, 1]
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(img1.device)
    return ssim(img1, img2)

def plot_metrics(train_mse, train_rewards, val_psnr, save_path="training_metrics.png"):
    """Generates and saves plots for training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training MSE and Reward
    plt.subplot(1, 2, 1)
    plt.plot(train_mse, label='MSE Loss', color='red', alpha=0.7)
    plt.plot(train_rewards, label='Reward', color='blue', alpha=0.7)
    plt.title('Training MSE and Reward per Step')
    plt.xlabel('Training Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Plot 2: Validation PSNR
    plt.subplot(1, 2, 2)
    plt.plot(val_psnr, label='Validation PSNR', color='green', marker='o')
    plt.title('Validation PSNR per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./Metrics_Results/"+save_path)
    logging.info(f"📈 Metrics plots saved to {save_path}")
    plt.close()

def validate(model, val_loader, device, architecture, epoch, output_dir="."):
    """
    Validates the model on the validation set.
    Handles both standard residual and U-Net architectures with skip connections (element-wise add).
    Assumes val_loader uses custom_collate_fn and yields batches as lists of (lr, hr) pairs.
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    val_img_dir = os.path.join(output_dir, "validation_images")
    os.makedirs(val_img_dir, exist_ok=True)
    saved_example = False

    with torch.no_grad():
        for batch in val_loader:
            for lr_img, hr_img in batch:
                lr_img = lr_img.unsqueeze(0).to(device)
                hr_img = hr_img.unsqueeze(0).to(device)

                features = model.initial_conv(lr_img)
                initial_features = features

                if model.UNet_initialization == 1:
                    feat_mat = []
                    for i in range(model.num_layers):
                        if i >= len(architecture):
                            break
                        action_idx = architecture[i]
                        if not (0 <= action_idx < len(model.layer_options)):
                            continue
                        chosen_layer = model.unet_layer_options[action_idx][i]

                        if i < model.num_layers // 2:
                            if not model.input_skip_only:
                                feat_mat.append(features.clone())
                            features = chosen_layer(features)
                            if i < len(model.downsamplers):
                                features = model.downsamplers[i](features)
                        
                        # Bottleneck layer (middle)
                        elif i == model.num_layers // 2:
                            features = chosen_layer(features)
                        
                        # Upsampling phase (second half)
                        else:
                            up_idx = i - (model.num_layers // 2 + 1)
                            skip = None
                            if not model.input_skip_only and feat_mat:
                                skip = feat_mat.pop()
                            if 0 <= up_idx < len(model.upsamplers):
                                features = model.upsamplers[up_idx](features)
                            features = chosen_layer(features)
                            if skip is not None:
                                if skip.shape[-2:] != features.shape[-2:]:
                                    raise ValueError(
                                        f"Spatial mismatch for skip connection at layer {i}: "
                                        f"features spatial {features.shape[-2:]}, skip spatial {skip.shape[-2:]}"
                                    )
                                if len(model.skip_combiners) == 0:
                                    raise ValueError(
                                        "Skip combiners are not initialized but a skip connection is available."
                                    )
                                comb_idx = min(up_idx, len(model.skip_combiners) - 1)
                                features = model.fuse_local_skip(features, skip, comb_idx)
                else:
                    # Standard residual path
                    for i in range(model.num_layers):
                        if i >= len(architecture):
                            break
                        
                        layer_idx = architecture[i]
                        if 0 <= layer_idx < len(model.layer_options):
                            chosen_layer = model.layer_options[layer_idx]
                            features = features + chosen_layer(features)
                # Fuse global skips before reconstruction (both paths)
                features = model.combine_with_global_skips(features, lr_img, initial_features)
                sr_img = model.upsampler(features)
                if sr_img.shape != lr_img.shape:
                    raise RuntimeError(
                        f"Validation output shape mismatch: pred {sr_img.shape}, input {lr_img.shape}"
                    )
                sr_img = model.output_activation(sr_img)
                sr_img = sr_img.clamp(0.0, 1.0)

                # Calculate PSNR (with numerical stability)
                mse = torch.mean((sr_img - hr_img) ** 2)
                mse = torch.clamp(mse, min=1e-10)  # Avoid log(0)
                psnr = 10 * torch.log10(1.0 / mse)

                # Calculate SSIM
                ssim = calculate_ssim(sr_img, hr_img)

                total_psnr += psnr.item()
                total_ssim += ssim.item()
                count += 1

                # Save first validation example
                if not saved_example:
                    # Convert to numpy (HWC, [0,1])
                    lr_vis = lr_img[0].cpu().numpy().transpose(1, 2, 0)
                    sr_vis = sr_img[0].cpu().numpy().transpose(1, 2, 0)
                    hr_vis = hr_img[0].cpu().numpy().transpose(1, 2, 0)

                    lr_vis = np.clip(lr_vis * 255, 0, 255).astype(np.uint8)
                    sr_vis = np.clip(sr_vis * 255, 0, 255).astype(np.uint8)
                    hr_vis = np.clip(hr_vis * 255, 0, 255).astype(np.uint8)

                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(lr_vis, cmap='gray')
                    plt.title('Low Dose Input')
                    plt.axis('off')

                    arch_str = "-".join(map(str, architecture)) if architecture else "N/A"

                    plt.subplot(1, 3, 2)
                    plt.imshow(sr_vis, cmap='gray')
                    plt.title(f'SR Output (Epoch {epoch+1})\nArch: {arch_str}')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(hr_vis, cmap='gray')
                    plt.title('Full Dose Target')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(val_img_dir, f'validation_example_epoch_{epoch+1}.png'),
                        bbox_inches='tight'
                    )
                    plt.close()
                    saved_example = True

    if count == 0:
        logging.warning("No validation images processed!")
        return 0.0

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    logging.info(f"Validation PSNR: {avg_psnr:.4f} dB | Validation SSIM: {avg_ssim:.4f}")
    return avg_psnr

def calculate_block_flops(model, lr_patch_size, num_features, device):
    """Calculates the FLOPs for each available block."""
    logger = logging.getLogger()
    dummy_input = torch.randn(1, num_features, lr_patch_size, lr_patch_size).to(device)
    flops_list = []
    logger.info("--- Calculating FLOPs for each block ---")
    for i, block in enumerate(model.layer_options):
        macs, _ = profile(block, inputs=(dummy_input,), verbose=False)
        flops = macs * 2
        flops_list.append(flops / 1e9) # GFLOPs
        logger.info(f"  Block {i} ({block.__class__.__name__}): {flops/1e9:.4f} GFLOPs")
    return torch.tensor(flops_list, device=device)

def calculate_block_parameters(model, device):
    """Calculates the number of parameters for each available block."""
    logger = logging.getLogger()
    params_list = []
    logger.info("\n--- Calculating Parameters for each block ---")
    for i, block in enumerate(model.layer_options):
        num_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
        params_list.append(num_params / 1e6) # In Millions
        logger.info(f"  Block {i} ({block.__class__.__name__}): {num_params:,} parameters")
    return torch.tensor(params_list, device=device)
