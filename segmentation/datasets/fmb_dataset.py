"""
FMB Dataset Loader — independent of MMSeg.

Dataset layout:
  {root}/
    train_easy_files.txt   # filenames like 00001.png
    train_hard_files.txt
    val_easy_files.txt
    val_hard_files.txt
    test_easy_files.txt
    test_hard_files.txt
    {split}/
      Visible/
        easy/  {filename}.png   # RGB
        hard/  {filename}.png
      Infrared/
        {filename}.png          # Thermal (3-ch RGB-stored grayscale, flat)
      Label/
        {filename}.png          # GT mask, uint8, values 1-14 (0=background→ignore_index=255)

14 classes (0-indexed):
  0 Road, 1 Sidewalk, 2 Building, 3 Traffic Light, 4 Traffic Sign,
  5 Vegetation, 6 Sky, 7 Person, 8 Car, 9 Truck, 10 Bus,
  11 Motorcycle, 12 Bicycle, 13 Pole
"""

import os.path as osp
import random

import numpy as np
from PIL import Image
from PIL import ImageFilter

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


CLASSES = [
    "Road", "Sidewalk", "Building", "Traffic Light", "Traffic Sign",
    "Vegetation", "Sky", "Person", "Car", "Truck", "Bus",
    "Motorcycle", "Bicycle", "Pole",
]
NUM_CLASSES = 14
IGNORE_INDEX = 255

# ImageNet stats for RGB
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]

# Thermal stored as 3-ch, normalize per-channel mean≈0.5
THM_MEAN = [0.5, 0.5, 0.5]
THM_STD  = [0.5, 0.5, 0.5]


class FMBDataset(Dataset):
    """RGB + Thermal semantic segmentation dataset for FMB benchmark."""

    CLASSES      = CLASSES
    NUM_CLASSES  = NUM_CLASSES
    IGNORE_INDEX = IGNORE_INDEX

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int = 512,
        augment: bool = True,
        eval_resize_mode: str = "stretch",
        train_resize_mode: str = "legacy",
        cat_max_ratio: float = 1.0,
        blur_prob: float = 0.0,
        photo_distort: bool = False,
    ):
        """
        Args:
            root:      Path to dataset root, e.g. /home/jl/dataset/FMB
            split:     One of 'train', 'val', 'test', 'trainval'
            crop_size: Square crop size used during training; evaluation
                       images are resized to (crop_size, crop_size).
            augment:   Enable data augmentation (only applies to 'train' or 'trainval').
            eval_resize_mode:
                - "stretch": direct resize to crop_size x crop_size (legacy behavior)
                - "letterbox": keep aspect ratio, then pad to crop_size x crop_size
            train_resize_mode:
                - "legacy": resize scaled image to at least crop_size before cropping
                - "mmseg": MMDetection/MMSeg-style random ratio resize, then pad if needed
            cat_max_ratio: Max dominant-class ratio in random crop (1.0 disables).
            blur_prob: Probability of applying Gaussian blur during training.
            photo_distort: Enable stronger photometric distortion.
        """
        assert split in ("train", "val", "test", "trainval"), f"Unknown split: {split}"
        assert eval_resize_mode in ("stretch", "letterbox"), (
            f"Unknown eval_resize_mode: {eval_resize_mode}"
        )
        assert train_resize_mode in ("legacy", "mmseg"), (
            f"Unknown train_resize_mode: {train_resize_mode}"
        )
        self.root      = root
        self.split     = split
        self.crop_size = crop_size
        self.augment   = augment and (split in ("train", "trainval"))
        self.eval_resize_mode = eval_resize_mode
        self.train_resize_mode = train_resize_mode
        self.cat_max_ratio = cat_max_ratio
        self.blur_prob = blur_prob
        self.photo_distort = photo_distort

        # For trainval, we'll load from both train and val directories
        self.splits_to_load = ["train", "val"] if split == "trainval" else [split]

        self.samples = self._load_samples()

    # ------------------------------------------------------------------
    # Sample list construction
    # ------------------------------------------------------------------

    def _load_samples(self):
        samples = []
        for split_name in self.splits_to_load:
            for subset in ("easy", "hard"):
                txt = osp.join(self.root, f"{split_name}_{subset}_files.txt")
                if not osp.exists(txt):
                    continue

                rgb_dir = osp.join(self.root, split_name, "Visible")
                thm_dir = osp.join(self.root, split_name, "Infrared")
                lbl_dir = osp.join(self.root, split_name, "Label")

                with open(txt) as f:
                    for line in f:
                        fname = line.strip()
                        if not fname:
                            continue
                        samples.append(
                            dict(
                                rgb=osp.join(rgb_dir, subset, fname),
                                thm=osp.join(thm_dir, fname),
                                lbl=osp.join(lbl_dir, fname),
                            )
                        )
        assert len(samples) > 0, (
            f"No samples found for split '{self.split}' in {self.root}"
        )
        return samples

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        rgb   = Image.open(s["rgb"]).convert("RGB")
        thm   = Image.open(s["thm"]).convert("RGB")  # grayscale stored as RGB
        label = Image.open(s["lbl"])                  # mode 'L', values 1-14 (0=background)

        if self.augment:
            rgb, thm, label = self._train_transform(rgb, thm, label)
        else:
            rgb, thm, label = self._eval_transform(rgb, thm, label)

        rgb_t = TF.normalize(TF.to_tensor(rgb), RGB_MEAN, RGB_STD)
        thm_t = TF.normalize(TF.to_tensor(thm), THM_MEAN, THM_STD)

        # FMB labels are 1-indexed (1=Road ... 14=Pole), 0=background.
        # Keep padded ignore (255) intact.
        lbl_np = np.array(label, dtype=np.int64)
        ignore_mask = (lbl_np == 0) | (lbl_np == IGNORE_INDEX)
        lbl_np = lbl_np - 1                   # 1-14 -> 0-13
        lbl_np[ignore_mask] = IGNORE_INDEX    # background/padding -> 255
        # Safety clamp for any unexpected labels
        lbl_np[(lbl_np < 0) | (lbl_np >= NUM_CLASSES)] = IGNORE_INDEX
        lbl_t = torch.from_numpy(lbl_np)

        return rgb_t, thm_t, lbl_t

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _train_transform(self, rgb, thm, label):
        # 1. Random scale. The legacy path forces both dimensions to be at
        #    least crop_size before cropping. The mmseg path mirrors the FMB
        #    configs used by MM SAM-Adapter: resize by ratio first, then pad
        #    only if the random crop is larger than the resized image.
        scale = random.uniform(0.5, 2.0)
        if self.train_resize_mode == "legacy":
            new_h = max(self.crop_size, int(rgb.height * scale))
            new_w = max(self.crop_size, int(rgb.width * scale))
        else:
            new_h = max(1, int(round(rgb.height * scale)))
            new_w = max(1, int(round(rgb.width * scale)))
        rgb   = TF.resize(rgb,   [new_h, new_w], interpolation=Image.BILINEAR)
        thm   = TF.resize(thm,   [new_h, new_w], interpolation=Image.BILINEAR)
        label = TF.resize(label, [new_h, new_w], interpolation=Image.NEAREST)

        # 1.5 Random Gaussian blur (baseline-style robustness augmentation)
        if self.blur_prob > 0.0 and random.random() < self.blur_prob:
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=1.0))
            thm = thm.filter(ImageFilter.GaussianBlur(radius=1.0))

        # 2. Pad to at least crop_size × crop_size
        rgb, thm, label = self._pad_if_needed(rgb, thm, label)

        # 3. Random crop
        i, j, h, w = self._sample_crop_params(label)
        rgb   = TF.crop(rgb,   i, j, h, w)
        thm   = TF.crop(thm,   i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # 4. Random horizontal flip
        if random.random() > 0.5:
            rgb, thm, label = TF.hflip(rgb), TF.hflip(thm), TF.hflip(label)

        # 5. Photometric distortion (optionally stronger, baseline-style)
        if self.photo_distort:
            rgb, thm = self._photometric_distort(rgb, thm)
        else:
            # Legacy mild jitter on RGB only
            if random.random() > 0.5:
                rgb = TF.adjust_brightness(rgb, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                rgb = TF.adjust_contrast(rgb, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                rgb = TF.adjust_saturation(rgb, random.uniform(0.8, 1.2))

        return rgb, thm, label

    def _eval_transform(self, rgb, thm, label):
        sz = self.crop_size
        if self.eval_resize_mode == "stretch":
            rgb   = TF.resize(rgb,   [sz, sz], interpolation=Image.BILINEAR)
            thm   = TF.resize(thm,   [sz, sz], interpolation=Image.BILINEAR)
            label = TF.resize(label, [sz, sz], interpolation=Image.NEAREST)
        else:
            rgb, thm, label = self._letterbox_resize(rgb, thm, label, sz)
        return rgb, thm, label

    def _pad_if_needed(self, rgb, thm, label):
        w, h = rgb.size
        pad_h = max(0, self.crop_size - h)
        pad_w = max(0, self.crop_size - w)
        if pad_h == 0 and pad_w == 0:
            return rgb, thm, label
        # TF.pad padding order: (left, top, right, bottom)
        padding = (0, 0, pad_w, pad_h)
        rgb   = TF.pad(rgb,   padding, fill=0)
        thm   = TF.pad(thm,   padding, fill=0)
        label = TF.pad(label, padding, fill=IGNORE_INDEX)
        return rgb, thm, label

    def _sample_crop_params(self, label):
        """Sample random crop params with optional cat_max_ratio constraint."""
        crop_hw = (self.crop_size, self.crop_size)
        if self.cat_max_ratio >= 1.0:
            return T.RandomCrop.get_params(label, crop_hw)

        last = None
        for _ in range(10):
            i, j, h, w = T.RandomCrop.get_params(label, crop_hw)
            last = (i, j, h, w)
            lbl_crop = np.array(TF.crop(label, i, j, h, w), dtype=np.int64)
            # Convert 1..14 -> 0..13 and ignore 0-background
            lbl_crop -= 1
            lbl_crop[lbl_crop < 0] = IGNORE_INDEX
            valid = lbl_crop != IGNORE_INDEX
            if valid.sum() == 0:
                continue
            cnt = np.bincount(lbl_crop[valid].reshape(-1), minlength=NUM_CLASSES)
            dominant = cnt.max() / cnt.sum()
            if dominant < self.cat_max_ratio:
                return i, j, h, w
        return last

    def _photometric_distort(self, rgb, thm):
        """A stronger color/intensity distortion similar to MMSeg default setup."""
        # random brightness
        if random.random() < 0.5:
            factor = random.uniform(0.875, 1.125)
            rgb = TF.adjust_brightness(rgb, factor)
            thm = TF.adjust_brightness(thm, factor)

        # contrast can be before or after color transforms
        contrast_first = random.random() < 0.5
        if contrast_first and random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            rgb = TF.adjust_contrast(rgb, factor)
            thm = TF.adjust_contrast(thm, factor)

        # saturation/hue only affect RGB semantics
        if random.random() < 0.5:
            rgb = TF.adjust_saturation(rgb, random.uniform(0.5, 1.5))
        if random.random() < 0.5:
            rgb = TF.adjust_hue(rgb, random.uniform(-0.1, 0.1))

        if (not contrast_first) and random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            rgb = TF.adjust_contrast(rgb, factor)
            thm = TF.adjust_contrast(thm, factor)

        return rgb, thm

    def _letterbox_resize(self, rgb, thm, label, target_size):
        """Keep aspect ratio, then pad to square target size."""
        ow, oh = rgb.size
        scale = min(target_size / max(ow, 1), target_size / max(oh, 1))
        nw = max(1, int(round(ow * scale)))
        nh = max(1, int(round(oh * scale)))

        rgb = TF.resize(rgb, [nh, nw], interpolation=Image.BILINEAR)
        thm = TF.resize(thm, [nh, nw], interpolation=Image.BILINEAR)
        label = TF.resize(label, [nh, nw], interpolation=Image.NEAREST)

        pad_w = target_size - nw
        pad_h = target_size - nh
        padding = (0, 0, pad_w, pad_h)  # pad right/bottom
        rgb = TF.pad(rgb, padding, fill=0)
        thm = TF.pad(thm, padding, fill=0)
        label = TF.pad(label, padding, fill=IGNORE_INDEX)
        return rgb, thm, label


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def build_fmb_dataloaders(
    root: str,
    crop_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    val_batch_size: int = 1,
):
    """Return (train_loader, val_loader, test_loader)."""
    from torch.utils.data import DataLoader

    train_ds = FMBDataset(root, split="train", crop_size=crop_size, augment=True)
    val_ds   = FMBDataset(root, split="val",   crop_size=crop_size, augment=False)
    test_ds  = FMBDataset(root, split="test",  crop_size=crop_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
