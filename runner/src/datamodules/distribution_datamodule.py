import math
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Sampler, TensorDataset, random_split
from torchdyn.datasets import ToyDataset

from src import utils

from .components.base import BaseLightningDataModule
from .components.time_dataset import load_dataset
from .components.tnet_dataset import SCData
from .components.two_dim import data_distrib
from .components.breizhsr_dataset import BreizshSRDataset
from .components.PastisSR import PASTISSR
from .components.sen2naip import Sen2NAIPDataset
from .components.transform import SRNormalize, Identity
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import os

import random

log = utils.get_pylogger(__name__)


class BreizhSR(LightningDataModule):
    """
    BreizhSR

    """
    pass_to_model = True
    IS_TRAJECTORY = False


    def __init__(
        self,
        sen2_amount: int,
        ):
        super().__init__() 
        self.dims = (4, 296, 296)
        self.sen2_amount = sen2_amount
        self.generator = torch.Generator()
        
    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later

        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
      
        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        path="/share/projects/sesure/aimi/data/BreizhSR_RGBNIR"
        if stage == "fit" or stage is None:
            self.train =  BreizshSRDataset(
                dataset_root=path,
                split="train",
                sen2_amount=self.sen2_amount)

            self.validate =  BreizshSRDataset(
                dataset_root=path,
                split="val",
                sen2_amount=self.sen2_amount)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = BreizshSRDataset(
                dataset_root=path,
                split="test",
                sen2_amount=self.sen2_amount)
            #self.test = SingleImageDataset("/share/home/okabayas/draft/team2024.pt")#("/share/home/okabayas/draft/paul.pt")
    def on_train_epoch_start(self):
        seed = torch.randint(0, 10000, (1,)).item()  
        self.generator.manual_seed(seed)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=8, shuffle=True, pin_memory=True, generator=self.generator)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        #import ipdb; ipdb.set_trace()
        return DataLoader(self.test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
        


class Pastis_SR_Individual(LightningDataModule):
    """
    PASTIS SR Individual - Single S2 image super-resolution dataset
    """
    pass_to_model = True
    IS_TRAJECTORY = False

    def __init__(
        self,
        sen2_amount: int = 1,
        nb_split: int = 1,
        ):
        super().__init__() 
        self.dims = (4, 128, 128)  # RGB + NIR, adjusted for split images
        self.sen2_amount = sen2_amount
        self.nb_split = nb_split
        self.generator = torch.Generator()
        
    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later

        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
        from .components.PastisSR import PASTIS_SR_INDIVIDUAL as PASTIS_SR_INDIVIDUAL_Dataset
        
        # Assign train/val datasets for use in dataloaders
        path = "/share/projects/ottopia/superstable/data/pastis/PASTIS-HD"
        norm_path = "/share/projects/ottopia/superstable/data/pastis/PASTIS-HD"

        if stage == "fit" or stage is None:
            self.train = PASTIS_SR_INDIVIDUAL_Dataset(
                path=path,
                folds=[1, 2, 3],  # Training folds
                nb_split=self.nb_split,
                norm_path=norm_path,
            )
            self.validate = PASTIS_SR_INDIVIDUAL_Dataset(
                path=path,
                folds=[4],  # Validation fold
                nb_split=self.nb_split,
                norm_path=norm_path,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = PASTIS_SR_INDIVIDUAL_Dataset(
                path=path,
                folds=[5],  # Test fold
                nb_split=self.nb_split,
                norm_path=norm_path,
            )

    def on_train_epoch_start(self):
        seed = torch.randint(0, 10000, (1,)).item()  
        self.generator.manual_seed(seed)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=16, num_workers=8, shuffle=True, pin_memory=True, generator=self.generator)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)


class PastisSR(LightningDataModule):
    """
    Patis + Pastis HD

    """
    pass_to_model = True
    IS_TRAJECTORY = False


    def __init__(
        self,
        sen2_amount: int,
        ):
        super().__init__() 
        self.dims = (4, 296, 296)
        self.sen2_amount = sen2_amount
        self.generator = torch.Generator()
        self.normalize = SRNormalize()
        
    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later

        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
      
        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        path = "/share/projects/ottopia/superstable/data/pastis/PASTIS-HD"

        if stage == "fit" or stage is None:
            self.train =  PASTISSR(
                            path = "/share/projects/ottopia/superstable/data/pastis/PASTIS-HD",
                            modalities = ["spot", "s2-multi"],
                            transform = self.normalize,
                            split="train",
                            super_res = True,
                            temporal_split=30,
                            )
            self.validate =  PASTISSR(
                                path = "/share/projects/ottopia/superstable/data/pastis/PASTIS-HD",
                                modalities = ["spot", "s2-multi"],
                                transform = self.normalize,
                                split="val",
                                super_res = True,
                                temporal_split=30,
                            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = PASTISSR(
                                path = "/share/projects/ottopia/superstable/data/pastis/PASTIS-HD",
                                modalities = ["spot", "s2-multi"],
                                transform = self.normalize,
                                split="train",
                                super_res = True,
                                temporal_split=5,
                            )

    def on_train_epoch_start(self):
        seed = torch.randint(0, 10000, (1,)).item()  
        self.generator.manual_seed(seed)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=1, num_workers=8, shuffle=True, pin_memory=True, generator=self.generator)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        


class Sen2NAIP(LightningDataModule):
    """
    Sen2NAIP
    """
    pass_to_model = True
    IS_TRAJECTORY = False


    def __init__(
        self,
        sen2_amount: int,
        ):
        super().__init__() 
        self.dims = (4, 520, 520)
        self.sen2_amount = sen2_amount
        self.generator = torch.Generator()
        
    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later

        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
      
        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        path = "/share/projects/sesure/aimi/data/sen2naip/mini/"

        if stage == "fit" or stage is None:
            self.train =  Sen2NAIPDataset(taco_folder=path, split="train")
            self.validate =  Sen2NAIPDataset(taco_folder=path, split="val")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = Sen2NAIPDataset(taco_folder=path, split="test")

    def on_train_epoch_start(self):
        seed = torch.randint(0, 10000, (1,)).item()  
        self.generator.manual_seed(seed)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=8, num_workers=8, shuffle=True, pin_memory=True, generator=self.generator)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        #import ipdb; ipdb.set_trace()
        return DataLoader(self.test, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)