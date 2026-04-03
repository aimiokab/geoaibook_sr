import os
import torch
import importlib

from functools import lru_cache
from torch.utils.data import Dataset


@lru_cache(maxsize=10)
def cached_torch_load(filename):
    return torch.load(filename)


def load_fun(fullname):
    path, name = fullname.rsplit('.', 1)
    return getattr(importlib.import_module(path), name)


class Sen2VenusDataset(Dataset):
    def __init__(self, cfg, is_training=True):
        self.root_path = cfg.dataset.root_path
        self.relname = 'train'
        if not is_training:
            self.relname = 'test'
        self.fdir = os.path.join(self.root_path, self.relname)
        self._load_files()
        self._filter_files(cfg)

    def _filter_files(self, cfg):
        places = cfg.dataset.get('places')
        if places is not None and not places == []:
            self.files = list(filter(
                lambda name: name.lower().split('_')[1] in places,
                self.files))

    def _load_files(self):
        print('load {} files from {}'.format(self.relname, self.fdir))
        self.files = []
        for dirpath, dirs, files in os.walk(self.fdir):
            for filename in files:
                if filename.endswith('.pt'):
                    self.files.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return cached_torch_load(self.files[index])