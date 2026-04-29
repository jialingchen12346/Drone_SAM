"""Generic paired external unlabeled dataset for label-efficient training.

The dataset is intentionally label-free. It returns an all-ignore target so it
can be concatenated with the existing FMB unlabeled subset without changing the
training loop contract.
"""

from __future__ import annotations

import json
import os.path as osp
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from PIL import ImageFilter

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from segmentation.datasets.fmb_dataset import (
    IGNORE_INDEX,
    RGB_MEAN,
    RGB_STD,
    THM_MEAN,
    THM_STD,
)


class ExternalPairedUnlabeledDataset(Dataset):
    """Load RGB/aux pairs from a manifest.

    Manifest formats:
      {"root": "/data/MFNet", "samples": [{"id": "mfnet/00001D", "rgba": "images/00001D.png"}]}
      [{"id": "x", "rgb": "/abs/rgb.png", "aux": "/abs/thermal.png"}]

    A record with ``rgba`` is split into RGB = first three channels and aux =
    alpha channel replicated to RGB. This matches the common MFNet RGB-T PNG
    packaging used by several repos.
    """

    def __init__(
        self,
        index_path: str,
        crop_size: int = 512,
        augment: bool = False,
        blur_prob: float = 0.0,
        photo_distort: bool = False,
    ):
        self.index_path = index_path
        self.crop_size = crop_size
        self.augment = augment
        self.blur_prob = blur_prob
        self.photo_distort = photo_distort

        with open(index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            self.root = payload.get("root", "")
            self.samples: List[Dict[str, Any]] = payload.get("samples", [])
        elif isinstance(payload, list):
            self.root = ""
            self.samples = payload
        else:
            raise ValueError(f"Unsupported external index format: {index_path}")

        if not self.samples:
            raise ValueError(f"No samples found in external index: {index_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, path: str) -> str:
        if osp.isabs(path):
            return path
        return osp.join(self.root, path)

    def _load_pair(self, rec: Dict[str, Any]):
        if "rgba" in rec:
            rgba = Image.open(self._resolve(rec["rgba"])).convert("RGBA")
            r, g, b, a = rgba.split()
            rgb = Image.merge("RGB", (r, g, b))
            aux = Image.merge("RGB", (a, a, a))
            return rgb, aux

        rgb_key = "rgb" if "rgb" in rec else "image"
        aux_key = "aux" if "aux" in rec else "thermal"
        if rgb_key not in rec or aux_key not in rec:
            raise KeyError(f"External sample requires rgba or rgb+aux fields: {rec}")
        rgb = Image.open(self._resolve(rec[rgb_key])).convert("RGB")
        aux = Image.open(self._resolve(rec[aux_key])).convert("RGB")
        return rgb, aux

    def _pad_if_needed(self, rgb, aux):
        w, h = rgb.size
        pad_h = max(0, self.crop_size - h)
        pad_w = max(0, self.crop_size - w)
        if pad_h == 0 and pad_w == 0:
            return rgb, aux
        padding = (0, 0, pad_w, pad_h)
        return TF.pad(rgb, padding, fill=0), TF.pad(aux, padding, fill=0)

    def _train_transform(self, rgb, aux):
        scale = random.uniform(0.5, 2.0)
        new_h = max(self.crop_size, int(rgb.height * scale))
        new_w = max(self.crop_size, int(rgb.width * scale))
        rgb = TF.resize(rgb, [new_h, new_w], interpolation=Image.BILINEAR)
        aux = TF.resize(aux, [new_h, new_w], interpolation=Image.BILINEAR)

        if self.blur_prob > 0.0 and random.random() < self.blur_prob:
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=1.0))
            aux = aux.filter(ImageFilter.GaussianBlur(radius=1.0))

        rgb, aux = self._pad_if_needed(rgb, aux)
        i = random.randint(0, rgb.height - self.crop_size)
        j = random.randint(0, rgb.width - self.crop_size)
        rgb = TF.crop(rgb, i, j, self.crop_size, self.crop_size)
        aux = TF.crop(aux, i, j, self.crop_size, self.crop_size)

        if random.random() > 0.5:
            rgb, aux = TF.hflip(rgb), TF.hflip(aux)

        if self.photo_distort:
            if random.random() < 0.5:
                rgb = TF.adjust_brightness(rgb, random.uniform(0.875, 1.125))
            if random.random() < 0.5:
                rgb = TF.adjust_contrast(rgb, random.uniform(0.5, 1.5))
            if random.random() < 0.5:
                rgb = TF.adjust_saturation(rgb, random.uniform(0.5, 1.5))
        return rgb, aux

    def _eval_transform(self, rgb, aux):
        sz = self.crop_size
        rgb = TF.resize(rgb, [sz, sz], interpolation=Image.BILINEAR)
        aux = TF.resize(aux, [sz, sz], interpolation=Image.BILINEAR)
        return rgb, aux

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        rgb, aux = self._load_pair(rec)
        if self.augment:
            rgb, aux = self._train_transform(rgb, aux)
        else:
            rgb, aux = self._eval_transform(rgb, aux)

        rgb_t = TF.normalize(TF.to_tensor(rgb), RGB_MEAN, RGB_STD)
        aux_t = TF.normalize(TF.to_tensor(aux), THM_MEAN, THM_STD)
        gt = torch.full((self.crop_size, self.crop_size), IGNORE_INDEX, dtype=torch.long)
        sample_id = rec.get("id", f"external/{Path(rec.get('rgba', rec.get('rgb', str(idx)))).stem}")
        return rgb_t, aux_t, gt, sample_id
