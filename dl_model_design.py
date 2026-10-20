import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from pydicom import dcmread
except Exception:  # pragma: no cover - optional dependency
    dcmread = None
from torchvision import transforms
from torchvision.transforms import functional as TF

from blocks_2 import MetaBlock, ResBlock, MambaBlock, downsampling_block, upsampling_block, GNNBlock, AttentionBlock, ConvLSTMBlock, ConvLSTMCell
from dataset import ImageDataset, PairedTransform, Updated_ImageDataset


RESULTS_DIR = "dl_model_design_results_4"
NORM_RANGE_MIN = -1024.0
NORM_RANGE_MAX = 3072.0
TRUNCATION_MIN = -600.0
TRUNCATION_MAX = 1400.0


class Unet_block_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.block_1 = MetaBlock(operation=AttentionBlock(128, window_size=8), num_features=128)
        self.downsample_1 = downsampling_block(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.block_2 = MetaBlock(operation=AttentionBlock(256, window_size=8), num_features=256)
        self.downsample_2 = downsampling_block(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)

        self.block_3 = MetaBlock(operation=ConvLSTMBlock(512), num_features=512)
        self.downsample_3 = downsampling_block(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)

        self.block_4 = MetaBlock(operation=ConvLSTMBlock(1024), num_features=1024)

        self.upsample_3 = upsampling_block(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.block_5 = MetaBlock(operation=ConvLSTMBlock(512), num_features=512)
        self.upsample_2 = upsampling_block(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.block_6 = MetaBlock(operation=ConvLSTMBlock(256), num_features=256)
        self.upsample_1 = upsampling_block(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.block_7 = MetaBlock(operation=ConvLSTMBlock(128), num_features=128)

        self.skip_combiners = nn.ModuleList([
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 128, kernel_size=1),
        ])
        self.input_skip_proj = nn.Conv2d(1, 128, kernel_size=1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_lr):
        x = self.initial_conv(x_lr)

        x1 = self.block_1(x)
        xd1 = self.downsample_1(x1)
        x2 = self.block_2(xd1)
        xd2 = self.downsample_2(x2)
        x3 = self.block_3(xd2)
        xd3 = self.downsample_3(x3)
        x4 = self.block_4(xd3)

        xu3 = self.upsample_3(x4)
        x5 = self.block_5(xu3)
        x5_cat = self.skip_combiners[0](torch.cat([x5, xd2], dim=1))

        xu2 = self.upsample_2(x5_cat)
        x6 = self.block_6(xu2)
        x6_cat = self.skip_combiners[1](torch.cat([x6, xd1], dim=1))
        xu1 = self.upsample_1(x6_cat)
        x7 = self.block_7(xu1)
        x7_cat = self.skip_combiners[2](torch.cat([x7, x], dim=1))
        x7_cat = x7_cat + self.input_skip_proj(x_lr)

        y_pred = self.upsampler(x7_cat)
        if y_pred.shape != x_lr.shape:
            raise RuntimeError("Output shape does not match input shape")
        return y_pred


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse <= 0:
        return float("inf")
    data_range = 1
    if data_range <= 0:
        data_range = torch.tensor(1.0, device=img2.device)
    return 20 * torch.log10(data_range / torch.sqrt(mse))

def calulate_rmse(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse

def get_ssim_metric(device):
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
    except Exception as exc:  # pragma: no cover - fallback for missing dependency
        raise RuntimeError("torchmetrics is required for SSIM calculation") from exc

    if not hasattr(get_ssim_metric, "_cache"):
        get_ssim_metric._cache = {}

    key = str(device)
    if key not in get_ssim_metric._cache:
        get_ssim_metric._cache[key] = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    return get_ssim_metric._cache[key]


def calculate_ssim(img1, img2, ssim_metric=None):
    metric = ssim_metric if ssim_metric is not None else get_ssim_metric(img1.device)

    data_range = torch.max(img2) - torch.min(img2)
    if data_range <= 0:
        data_range = torch.tensor(1.0, device=img2.device)

    metric.data_range = float(data_range.item())
    metric.reset()
    value = metric(img1, img2)
    metric.reset()
    return value


def denormalize_and_truncate(image):
    """Convert normalized test images to HU values and apply the CT display range."""
    if image.numel() > 0 and image.detach().amin() >= 0 and image.detach().amax() <= 1:
        image = image * (NORM_RANGE_MAX - NORM_RANGE_MIN) + NORM_RANGE_MIN
    return torch.clamp(image, min=TRUNCATION_MIN, max=TRUNCATION_MAX)


def normalize_for_display(array):
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return arr

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr, dtype=np.float32)

    finite_values = arr[finite_mask]
    lower = float(finite_values.min())
    arr = arr - lower
    upper = float(arr[finite_mask].max())
    if upper <= 0:
        return np.zeros_like(arr, dtype=np.float32)

    arr = arr / upper
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


class NoCropPairedTransform:
    """
    Validation/test transform that returns (lr_tensor, hr_tensor) without random
    cropping or augmentation. Always returns tensors produced by ``transforms.ToTensor()``
    so values are in a consistent 0..1 range and shapes are (C,H,W).
    """
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def _ensure_2d(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            # For multi-slice arrays, keep a deterministic first slice.
            return arr[0]
        raise ValueError(f"Expected 2D image data, got shape {arr.shape}")

    def _to_array(self, img):
        if isinstance(img, np.ndarray):
            return self._ensure_2d(img)
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy().astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            return self._ensure_2d(arr)
    def __call__(self, qd_image, fd_image):
        # Convert inputs to numpy arrays
        qd_arr = self._to_array(qd_image)
        fd_arr = self._to_array(fd_image)

        qd_t = torch.from_numpy(qd_arr.astype(np.float32)).unsqueeze(0)
        fd_t = torch.from_numpy(fd_arr.astype(np.float32)).unsqueeze(0)

        # If sizes differ, resize LR to match HR using torchvision functional resize
        if qd_t.shape[-2:] != fd_t.shape[-2:]:
            qd_t = TF.resize(qd_t, list(fd_t.shape[-2:]), interpolation=transforms.InterpolationMode.BICUBIC)



        return qd_t, fd_t

def hybrid_loss(img_pred, img_target, mse_weight=0.8, ssim_weight=0.2, ssim_metric=None):
    mse_raw = torch.mean((img_pred - img_target) ** 2)
    data_range = torch.max(img_target) - torch.min(img_target)
    if data_range <= 0:
        data_range = torch.tensor(1.0, device=img_target.device)
    mse_value = mse_raw / (data_range ** 2)
    ssim_value = calculate_ssim(img_pred, img_target, ssim_metric=ssim_metric)
    return (mse_weight * mse_value) + (ssim_weight * (1.0 - ssim_value))


def log_message(message, log_path=None):
    print(message)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")



def plot_metrics(train_loss_history, val_psnr_history, val_ssim_history, val_rmse_history, output_dir=RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, color="red", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_psnr_history, color="green", marker="o", label="Validation PSNR")
    plt.title("Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_psnr.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_ssim_history, color="purple", marker="o", label="Validation SSIM")
    plt.title("Validation SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_ssim.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_rmse_history, color="orange", marker="o", label="Validation RMSE")
    plt.title("Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_rmse.png"), dpi=150)
    plt.close()



def save_validation_example(lr_img, sr_img, hr_img, epoch, val_psnr, val_ssim, val_rmse, output_dir=RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    lr_img = lr_img.detach().cpu().numpy().squeeze(0)
    sr_img = sr_img.detach().cpu().numpy().squeeze(0)
    hr_img = hr_img.detach().cpu().numpy().squeeze(0)


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(lr_img, cmap="gray")
    axes[0].set_title("Validation LR")
    axes[0].axis("off")

    axes[1].imshow(sr_img, cmap="gray")
    axes[1].set_title(
        f"Validation SR (Epoch {epoch + 1})"
    )
    axes[1].axis("off")

    axes[2].imshow(hr_img, cmap="gray")
    axes[2].set_title("Validation HR")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"validation_example_epoch_{epoch + 1}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate(model, dataloader, device, epoch=None, save_example=False, ssim_metric=None, postprocess=False):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    count = 0
    selected_example = None
    seen_samples = 0

    with torch.no_grad():
        for lr_img, hr_img in dataloader:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)
            sr_img = model(lr_img)

            if postprocess:
                sr_img = denormalize_and_truncate(sr_img)
                hr_img = denormalize_and_truncate(hr_img)

            batch_count = lr_img.size(0)
            if save_example and epoch is not None and batch_count > 0:
                seen_samples += batch_count
                if selected_example is None or random.randint(1, seen_samples) <= batch_count:
                    sample_idx = random.randrange(batch_count)
                    selected_example = (
                        lr_img[sample_idx].detach().cpu(),
                        sr_img[sample_idx].detach().cpu(),
                        hr_img[sample_idx].detach().cpu(),
                    )

            total_psnr += calculate_psnr(sr_img, hr_img).item()
            total_ssim += calculate_ssim(sr_img, hr_img, ssim_metric=ssim_metric).item()
            total_rmse += calulate_rmse(sr_img, hr_img).item()
            count += 1

    if count == 0:
        return 0.0, 0.0, 0.0

    if save_example and epoch is not None and selected_example is not None:
        save_validation_example(*selected_example, epoch=epoch, val_psnr=total_psnr / count, val_ssim=total_ssim / count, val_rmse=total_rmse / count)

    return total_psnr / count, total_ssim / count, total_rmse / count


def train_model(
    train_root=r"C:\Users\k23pham\Desktop\npy_file_t_v_t\train",
    val_root=r"C:\Users\k23pham\Desktop\npy_file_t_v_t\val",
    test_root=r"C:\Users\k23pham\Desktop\npy_file_t_v_t\test",
    epochs=200,
    batch_size=4,
    lr=1e-4,
    patch_size=64,
    num_workers=6,
    validation_interval=10,
    seed=42,
    mse_weight=0.2,
    ssim_weight=0.8
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "dl_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Run started at {datetime.now().isoformat()}\n")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training, but no CUDA device is available.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    log_message(f"Using device: {device} ({gpu_name})", log_path)

    train_dataset = Updated_ImageDataset(mode = 'train', dataset = train_root, test_patient = "L506", transform=PairedTransform(patch_size))
    val_dataset = Updated_ImageDataset(mode = 'val', dataset = val_root, test_patient = "L333", transform=NoCropPairedTransform())
    test_dataset = Updated_ImageDataset(mode = 'test', dataset = test_root, test_patient = "L506", transform=NoCropPairedTransform())

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    model = Unet_block_model().to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    non_trainable_parameters = total_parameters - trainable_parameters
    log_message(
        f"Model parameters: total={total_parameters:,}, trainable={trainable_parameters:,}, "
        f"non-trainable={non_trainable_parameters:,}",
        log_path,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ssim_metric = get_ssim_metric(device)

    train_loss_history = []
    val_psnr_history = []
    val_ssim_history = []
    val_rmse_history = []
    test_psnr = 0.0
    test_ssim = 0.0
    test_rmse = 0.0

    # resume from checkpoint if present
    checkpoint_path = os.path.join(RESULTS_DIR, "dl_model_design_results.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state_dict", {}))
            opt_state = ckpt.get("optimizer_state_dict")
            if opt_state is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception:
                    # optimizer state may require CPU tensors or different param groups;
                    # attempt a best-effort restoration by ignoring incompatible parts
                    log_message("[Resume] Warning: failed to fully restore optimizer state", log_path)

            train_loss_history = ckpt.get("train_loss_history", train_loss_history)
            val_psnr_history = ckpt.get("val_psnr_history", val_psnr_history)
            val_ssim_history = ckpt.get("val_ssim_history", val_ssim_history)
            val_rmse_history = ckpt.get("val_rmse_history", val_rmse_history)
            # Older final checkpoints did not store ``epoch``; the loss history
            # still records how many epochs had completed.
            start_epoch = int(ckpt.get("epoch", len(train_loss_history)))
            if start_epoch < 0:
                start_epoch = 0
            if start_epoch >= epochs:
                log_message(
                    f"[Resume] Checkpoint already completed {start_epoch} epochs; skipping training and running test evaluation.",
                    log_path,
                )
                log_message(f"[Resume] Starting test evaluation for checkpoint at epoch {start_epoch}...", log_path)
                test_psnr, test_ssim, test_rmse = validate(
                    model,
                    test_loader,
                    device,
                    ssim_metric=ssim_metric,
                    postprocess=False,
                )
                log_message(f"Test set: psnr={test_psnr:.4f} dB, ssim={test_ssim:.4f}, rmse={test_rmse:.4f}", log_path)
            log_message(f"[Resume] Loaded checkpoint from epoch {start_epoch}", log_path)
        except Exception as exc:
            log_message(f"[Resume] Failed to load checkpoint: {exc}", log_path)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for lr_img, hr_img in progress:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            sr_img = model(lr_img)
            loss = hybrid_loss(
                sr_img,
                hr_img,
                mse_weight=mse_weight,
                ssim_weight=ssim_weight,
                ssim_metric=ssim_metric,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(batches, 1)
        train_loss_history.append(avg_loss)

        do_validate = ((epoch + 1) % validation_interval == 0) or (epoch == epochs - 1)
        if do_validate:
            log_message(f"[Validation] Starting validation for epoch {epoch + 1}...", log_path)
            val_psnr, val_ssim, val_rmse = validate(
                model,
                val_loader,
                device,
                epoch=epoch,
                save_example=True,
                ssim_metric=ssim_metric,
            )
            val_psnr_history.append(val_psnr)
            val_ssim_history.append(val_ssim)
            val_rmse_history.append(val_rmse)
            log_message(
                f"[Validation] Completed epoch {epoch + 1}: PSNR={val_psnr:.4f} dB, SSIM={val_ssim:.4f}, RMSE={val_rmse:.4f}",
                log_path,
            )

            log_message(
                f"Epoch {epoch + 1}: loss={avg_loss:.4f}, psnr={val_psnr:.4f} dB, ssim={val_ssim:.4f}, RMSE={val_rmse:.4f}",
                log_path,
            )

            checkpoint_path = os.path.join(RESULTS_DIR, "dl_model_design_results.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss_history": train_loss_history,
                    "val_psnr_history": val_psnr_history,
                    "val_ssim_history": val_ssim_history,
                    "val_rmse_history": val_rmse_history,
                    "latest_train_loss": avg_loss,
                    "latest_val_psnr": val_psnr,
                    "latest_val_ssim": val_ssim,
                    "latest_val_rmse": val_rmse,
                },
                checkpoint_path,
            )
            log_message(f"[Checkpoint] Saved: {checkpoint_path}", log_path)

            plot_metrics(train_loss_history, val_psnr_history, val_ssim_history, val_rmse_history, output_dir=RESULTS_DIR)
        else:
            log_message(f"Epoch {epoch + 1}: loss={avg_loss:.4f}", log_path)

    test_psnr, test_ssim, test_rmse = validate(
        model,
        test_loader,
        device,
        ssim_metric=ssim_metric,
        postprocess=False,
    )
    log_message(f"Test set: psnr={test_psnr:.4f} dB, ssim={test_ssim:.4f}, rmse={test_rmse:.4f}", log_path)

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss_history": train_loss_history,
            "val_psnr_history": val_psnr_history,
            "val_ssim_history": val_ssim_history,
            "val_rmse_history": val_rmse_history,       
            "test_psnr": test_psnr,
            "test_ssim": test_ssim,
            "test_rmse": test_rmse,
        },
        os.path.join(RESULTS_DIR, "dl_model_design_results.pth"),
    )
    return model, train_loss_history, val_psnr_history, val_ssim_history, val_rmse_history, test_psnr, test_ssim, test_rmse


if __name__ == "__main__":
    train_model()