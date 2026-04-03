import torch
import numpy as np
import rasterio as rio
from tacoreader import TortillaDataFrame
import tacoreader
import torch.nn.functional as F
import os

class Sen2NAIPDataset(torch.utils.data.Dataset):
    def __init__(self, taco_folder, split="train"):
        # Load the dataset once in memory
        self.dataset: TortillaDataFrame = tacoreader.load(os.path.join(taco_folder, f"mini_{split}.taco"))
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Cache the file paths to avoid redundant Parquet reads
        if idx not in self.cache:
            sample: TortillaDataFrame = self.dataset.read(idx)
            lr: str = sample.read(0)
            hr: str = sample.read(1)
            self.cache[idx] = (lr, hr)
        else:
            lr, hr = self.cache[idx]

        # Open the files and load data
        with rio.open(lr) as src, rio.open(hr) as dst:
            lr_data: np.ndarray = src.read()
            hr_data: np.ndarray = dst.read()
        lr = torch.tensor(lr_data.astype(np.float32))/10000
        hr = torch.tensor(hr_data.astype(np.float32))/10000
        _, new_height, new_width = hr.shape
        img_lr_up = F.interpolate(lr[None,...], (new_height, new_width), mode="bicubic").squeeze()
        dict_return = {
            'img_lr': 2*lr-1,
            'img_hr': 2*hr-1,
            'img_lr_up': 2*img_lr_up-1,
            'item_name': idx,
            'indexes': idx,
        }
        return dict_return