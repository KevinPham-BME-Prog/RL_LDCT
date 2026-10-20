from pydicom import dcmread
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np
import torch
import random
import os
import glob
import re

# Top-level callable class (picklable) that performs the paired transform.
class PairedTransform:
    """
    Always returns (lr_tensor, hr_tensor) with spatial size (patch_size, patch_size).
    Both tensors are single-channel (grayscale). Use this when you always want 64x64.
    """
    def __init__(self, patch_size=64, norm_range_min = None, norm_range_max = None):
        self.patch_size = int(patch_size)
        self.norm_range_min = norm_range_min
        self.norm_range_max = norm_range_max

    def _to_tensor(self, img):
        if isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img)
        elif torch.is_tensor(img):
            tensor = img
        else:
            raise TypeError("Unsupported image type for transform")

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim != 3:
            raise TypeError("Expected a 2D grayscale image or a 3D tensor")

        return tensor.float()

    def __call__(self, QD_image, FD_image):
        # ensure tensor
        QD = self._to_tensor(QD_image)
        FD = self._to_tensor(FD_image)

        # sample crop params from FD (reference) to get an HR patch of size patch_size x patch_size
        i, j, h, w = transforms.RandomCrop.get_params(FD, output_size=(self.patch_size, self.patch_size))
        QD_c = TF.crop(QD, i, j, h, w)
        FD_c = TF.crop(FD, i, j, h, w)

        # deterministic flips (same decision for both)
        if random.random() < 0.5:
            QD_c = TF.hflip(QD_c); FD_c = TF.hflip(FD_c)
        if random.random() < 0.5:
            QD_c = TF.vflip(QD_c); FD_c = TF.vflip(FD_c)

        # ensure LR is also exactly patch_size x patch_size (no scale change)
        #lr_image = TF.resize(QD_c, (self.patch_size, self.patch_size),
        #                     interpolation=transforms.InterpolationMode.BICUBIC)
        return QD_c, FD_c

def get_transforms(patch_size=64, norm_range_min=-1024.0, norm_range_max=3072.0):
    """
    Return a picklable callable for multiprocessing dataloaders that always
    yields (LR, HR) tensors of size (patch_size, patch_size).
    """
    return PairedTransform(patch_size, norm_range_min, norm_range_max)

def get_validation_transform(scale_factor): #not needed however needs to be adjusted in the tran code
    """
    Returns a paired transformation function for validation.
    It resizes the full HR image to create the LR image without random cropping.
    """
    def paired_transform(hr_image):
        lr_image = transforms.functional.resize(
            hr_image, 
            (hr_image.height // scale_factor, hr_image.width // scale_factor), 
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        return transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_image)
    return paired_transform


class ImageDataset(Dataset):
    """
    Paired dataset loader. Pass a single folder (root_dir) that either:
      - contains two subfolders "Quarter" and "Full" (recommended), or
      - contains QD/FD files in the same folder (filenames include 'qd'/'fd' or 'quarter'/'full').
    Pairs are matched by a normalized basename so 'case_QD_001.IMA' -> 'case_FD_001.IMA' will match.
    The provided transform is called as transform(QD_PIL, FD_PIL).
    """
    def __init__(self, root_dir, quarter_subdir="Quarter", full_subdir="Full", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        exts = (".ima", ".IMA", ".dcm", ".DCM")

        def is_image_file(p):
            return any(p.endswith(e) for e in exts)

        def normalize_key(name):
            # remove tokens and separators, lowercase for robust matching
            k = name.lower()
            k = re.sub(r"(qd|fd|quarter|full)", "", k)
            k = re.sub(r"[_\-\.\s]+", "", k)
            return k

        q_dir = os.path.join(root_dir, quarter_subdir)
        f_dir = os.path.join(root_dir, full_subdir)

        pairs = []

        if os.path.isdir(q_dir) and os.path.isdir(f_dir):
            q_files = sorted([p for p in glob.glob(os.path.join(q_dir, "*")) if is_image_file(p)])
            f_files = sorted([p for p in glob.glob(os.path.join(f_dir, "*")) if is_image_file(p)])

            f_by_key = {}
            for p in f_files:
                key = normalize_key(os.path.splitext(os.path.basename(p))[0])
                f_by_key.setdefault(key, []).append(p)

            for q in q_files:
                qkey = normalize_key(os.path.splitext(os.path.basename(q))[0])
                if qkey in f_by_key:
                    pairs.append((q, f_by_key[qkey][0]))

        else:
            # single-folder case: split candidates by tokens, then match by normalized key
            all_files = sorted([p for p in glob.glob(os.path.join(root_dir, "*")) if is_image_file(p)])
            q_candidates = []
            f_candidates = []
            for p in all_files:
                b = os.path.splitext(os.path.basename(p))[0].lower()
                if re.search(r"(qd|quarter)", b):
                    q_candidates.append(p)
                elif re.search(r"(fd|full)", b):
                    f_candidates.append(p)

            # Build maps and match by normalized key
            f_by_key = {}
            for p in f_candidates:
                key = normalize_key(os.path.splitext(os.path.basename(p))[0])
                f_by_key.setdefault(key, []).append(p)

            for q in q_candidates:
                qkey = normalize_key(os.path.splitext(os.path.basename(q))[0])
                if qkey in f_by_key:
                    pairs.append((q, f_by_key[qkey][0]))

            # final fallback: if counts equal and still no pairs, zip sorted lists
            if not pairs and len(q_candidates) == len(f_candidates) and len(q_candidates) > 0:
                pairs = list(zip(sorted(q_candidates), sorted(f_candidates)))

        if not pairs:
            raise RuntimeError(f"No paired Quarter/Full files found in {root_dir}")

        self.pairs = sorted(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qpath, fpath = self.pairs[idx]
        qds = dcmread(qpath); fds = dcmread(fpath)
        qarr = qds.pixel_array; farr = fds.pixel_array
        if self.transform:
            return self.transform(qarr, farr)

        q_tensor = torch.from_numpy(qarr).unsqueeze(0).float()
        f_tensor = torch.from_numpy(farr).unsqueeze(0).float()
        return q_tensor, f_tensor

def custom_collate_fn(batch):
    """
    Custom collate function for the validation dataloader.
    Since validation images can have different sizes, we don't stack them
    into a single tensor but return a list of (LR, HR) tuples.
    """
    return batch

class Updated_ImageDataset(Dataset):
    def __init__(self, mode, dataset, test_patient, transform=None):
        self.dataset = dataset
        self.transform = transform
        input_path = sorted(glob.glob(os.path.join(dataset, '*_input.npy')))
        target_path = sorted(glob.glob(os.path.join(dataset, '*_target.npy')))

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            self.input_path = input_
            self.target_path = target_
        elif mode == 'val':
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            self.input_path = input_
            self.target_path = target_
        elif mode == 'test':
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            self.input_path = input_
            self.target_path = target_
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'.")
    def __len__(self):
        return len(self.input_path)
    def __getitem__(self, index):
        input_image = np.load(self.input_path[index])
        target_image = np.load(self.target_path[index])
        if self.transform:
            return self.transform(input_image, target_image)
        else:
            return torch.from_numpy(input_image).unsqueeze(0).float(), torch.from_numpy(target_image).unsqueeze(0).float()