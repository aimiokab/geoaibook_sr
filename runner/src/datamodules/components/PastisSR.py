from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import geopandas as gpd
import torch
import rasterio
from datetime import datetime
import json
import torch.nn.functional as F
import math


def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name" and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    for key in ["s2", "s1-asc", "s1-des", "s1"]:
        if key in keys:
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor.float()
            keys.remove(key)
            key = '_'.join([key, "dates"])
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor.long()
            keys.remove(key)
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch]).float()
    return output

def prepare_dates(date_dict, reference_date):
    """Date formating."""
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return torch.tensor(d.values)

def split_image(image_tensor, nb_split, id):
    """
    Split the input image tensor into four quadrants based on the integer i.
    To use if Pastis data does not fit in your GPU memory.
    Returns the corresponding quadrant based on the value of i
    """
    if nb_split == 1:
        return image_tensor
    i1 = id // nb_split
    i2 = id % nb_split
    height, width = image_tensor.shape[-2:]
    half_height = height // nb_split
    half_width = width // nb_split
    if image_tensor.dim() == 4:
        return image_tensor[:, :, i1*half_height:(i1+1)*half_height, i2*half_width:(i2+1)*half_width].float()
    if image_tensor.dim() == 3:
        return image_tensor[:, i1*half_height:(i1+1)*half_height, i2*half_width:(i2+1)*half_width].float()
    if image_tensor.dim() == 2:
        return image_tensor[i1*half_height:(i1+1)*half_height, i2*half_width:(i2+1)*half_width].float()
    
def data_temporal_split(d, n_splits):
    items = sorted(d.items(), key=lambda x: int(x[0]))
    total = len(items)
    chunk_size = math.ceil(total / n_splits)
    return [dict(items[i:i + chunk_size]) for i in range(0, total, chunk_size)]

class PASTIS_SR_INDIVIDUAL(Dataset):
    def __init__(
        self,
        path,
        folds=None,
        nb_split=1,
        norm_path=None,
    ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            folds (list): list of folds to use
            nb_split (int): number of splits from one observation
            norm_path (str): path to the normalization stats
        """
        super().__init__()
        self.path = path
        self.nb_split = nb_split

        meta_patch = gpd.read_file(os.path.join(self.path, "metadata.geojson"))
        if folds is not None:
            meta_patch = meta_patch[meta_patch["Fold"].isin(folds)]

        # Create a list of individual samples (one per S2 image)
        samples = []
        for _, row in meta_patch.iterrows():
            patch_id = row['ID_PATCH']
            s2_dates = json.loads(row['dates-S2'])
            for s2_index in s2_dates.keys():
                samples.append({
                    'ID_PATCH': patch_id,
                    's2_index': int(s2_index),
                })
        self.samples = pd.DataFrame(samples)

        # Load normalization stats for S2
        self.norm = None
        if norm_path is not None:
            file_path = os.path.join(norm_path, "NORM_s2_patch.json")
            if os.path.exists(file_path):
                normvals = json.load(open(file_path))
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals[f"Fold_{f}"]["mean"] for f in selected_folds]
                stds = [normvals[f"Fold_{f}"]["std"] for f in selected_folds]
                s2_norm = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm = (
                    torch.from_numpy(s2_norm[0]).float(),
                    torch.from_numpy(s2_norm[1]).float(),
                )
            else:
                print(f"Warning: Normalization file not found at {file_path}. S2 data will not be normalized.")

    def __len__(self):
        return len(self.samples) * self.nb_split * self.nb_split

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "lr", "hr", "id"
        """
        # Determine which sample and which split part to load
        sample_info = self.samples.iloc[i // (self.nb_split * self.nb_split)]
        part = i % (self.nb_split * self.nb_split)
        patch_id = sample_info['ID_PATCH']
        s2_index = sample_info['s2_index']

        # Load S2 time series and select the specific image (LR)
        s2_ts_path = os.path.join(self.path, "DATA_S2", f"S2_{patch_id}.npy")
        s2_ts = torch.from_numpy(np.load(s2_ts_path)).to(torch.float32)
        # Select RGB and NIR bands (R, G, B, NIR)
        s2_image = s2_ts[s2_index, [2, 1, 0, 6], ...]
        s2_image = split_image(s2_image, self.nb_split, part)

        # Load corresponding SPOT image (HR)
        spot_path = os.path.join(self.path, 'DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019', f'SPOT6_RVB_1M00_2019_{patch_id}.tif')
        with rasterio.open(spot_path) as f:
            spot_image = torch.from_numpy(f.read()).to(torch.float32)
        spot_image = split_image(spot_image, self.nb_split, part)

        # Normalize S2 image
        if self.norm is not None:
            # Select norm stats for RGB and NIR bands
            norm_mean = self.norm[0][[2, 1, 0, 6]]
            norm_std = self.norm[1][[2, 1, 0, 6]]
            s2_image = (s2_image - norm_mean[:, None, None]) / norm_std[:, None, None]

        return {
            'lr': s2_image,
            'hr': spot_image,
            'id': f"{patch_id}_{s2_index}_{part}"
        }

class PASTISSR(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        folds=None,
        reference_date="2018-09-01",
        nb_split = 1,
        num_classes = 20,
        classif: bool = True,
        norm_path = None,
        split = "train",
        temporal_dropout = 0,
        super_res = False,
        temporal_split=2,
        ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            transform (torch module): transform to apply to the data
            folds (list): list of folds to use
            reference_date (str date): reference date for the data
            nb_split (int): number of splits from one observation
            num_classes (int): number of classes
        """
        super(PASTISSR, self).__init__()
        self.path = path
        self.transform = transform
        self.modalities = modalities
        self.nb_split = nb_split
        self.classif = classif
        self.split = split
        self.temporal_dropout = temporal_dropout
        self.super_res = super_res
        self.temporal_split = temporal_split


        assert split in ["train", "val", "test"], "Split must be train, val or test"
        if split == "train":
            folds = [1, 2, 3, 4, 5]
        elif split == "val":
            folds = [4]
        else:
            folds = [5]

        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.meta_patch = gpd.read_file(os.path.join(self.path, "metadata.geojson"))
        self.meta_patch['dates-S2']= self.meta_patch['dates-S2'].apply(lambda x: json.loads(x))
        self.meta_patch = (
            self.meta_patch.assign(dict_chunks=self.meta_patch['dates-S2'].apply(lambda d: data_temporal_split(d, self.temporal_split)))
            .explode('dict_chunks', ignore_index=True)
            .drop(columns='dates-S2')
            .rename(columns={'dict_chunks': 'dates-S2'})
            )
        self.meta_patch['dates-S2'] = self.meta_patch['dates-S2'].apply(lambda x: json.dumps(x))


        self.num_classes = num_classes

        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.norm = None

        if norm_path is not None:
            norm = {}
            for modality in self.modalities:
                file_path = os.path.join(norm_path, "NORM_{}_patch.json".format(modality))
                if not(os.path.exists(file_path)):
                    self.compute_norm_vals(norm_path, modality)
                normvals = json.load(open(file_path))
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                norm[modality] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                norm[modality] = (
                    torch.from_numpy(norm[modality][0]).float(),
                    torch.from_numpy(norm[modality][1]).float(),
                )
            self.norm = norm

        self.collate_fn = collate_fn


    def compute_norm_vals(self, folder, sat):
        norm_vals = {}
        for fold in range(1, 6):
            means = []
            stds = []
            for i, b in enumerate(self.meta_patch):
                data = self.__getitem__(i)[sat]
                if len(data.shape) == 4:
                    data = data.permute(1, 0, 2, 3)
                    means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
                    stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())
                else:
                    means.append(data.to(torch.float32).mean(dim=(1, 2)).numpy())
                    stds.append(data.to(torch.float32).std(dim=(1, 2)).numpy())

            mean = np.stack(means).mean(axis=0).astype(float)
            std = np.stack(stds).mean(axis=0).astype(float)

            norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """ 
        line = self.meta_patch.iloc[i // (self.nb_split * self.nb_split)]
        temporal_split = list(json.loads(line['dates-S2']).keys())
        name = line['ID_PATCH']
        part = i % (self.nb_split * self.nb_split)
        label = torch.from_numpy(np.load(os.path.join(self.path, 'ANNOTATIONS/TARGET_' + str(name) + '.npy'))[0].astype(np.int32))
        if self.classif:
            label = torch.unique(split_image(label, self.nb_split, part)).long()
            label = torch.sum(torch.nn.functional.one_hot(label, num_classes=self.num_classes), dim = 0)
            label = label[1:-1] #remove Background and Void classes
        else:
            label = label
        output = {'label': label, 'name': name}

        for modality in self.modalities:
            if modality == "spot":
                with rasterio.open(os.path.join(self.path, 'DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019/SPOT6_RVB_1M00_2019_' + str(name) + '.tif')) as f:
                    output["spot"] = split_image(torch.FloatTensor(f.read()), self.nb_split, part)
            elif modality == "s1-median":
                modality_name = "s1a"
                images = split_image(torch.from_numpy(np.load(os.path.join(
                            self.path,
                            "DATA_{}".format(modality_name.upper()),
                            "{}_{}.npy".format(modality_name.upper(), name),
                        ))), self.nb_split, part).to(torch.float32)
                out, _ = torch.median(images, dim = 0)
                output[modality] = out
            elif modality == "s1-mid":
                modality_name = "s1a"
                images = split_image(torch.from_numpy(np.load(os.path.join(
                            self.path,
                            "DATA_{}".format(modality_name.upper()),
                            "{}_{}.npy".format(modality_name.upper(), name),
                        ))), self.nb_split, part).to(torch.float32)
                out = images[images.shape[0]//2]
                output[modality] = out 
            elif modality == "s2-median":
                modality_name = "s2"
                images = split_image(torch.from_numpy(np.load(os.path.join(
                            self.path,
                            "DATA_{}".format(modality_name.upper()),
                            "{}_{}.npy".format(modality_name.upper(), name),
                        ))), self.nb_split, part).to(torch.float32)
                out, _ = torch.median(images, dim = 0) # mediane 
                output[modality] = out
            elif modality == "s2-mid":
                modality_name = "s2"
                images = split_image(torch.from_numpy(np.load(os.path.join(
                            self.path,
                            "DATA_{}".format(modality_name.upper()),
                            "{}_{}.npy".format(modality_name.upper(), name),
                        ))), self.nb_split, part).to(torch.float32)
                out = images[images.shape[0]//2] # image au milieu
                if self.super_res:
                    output[modality] = out[[2,1,0,6], ...]
                else:
                    output[modality] = out
            elif modality == "s1-4season-median":
                modality_name = "s1a"
                images = split_image(torch.from_numpy(np.load(os.path.join(
                            self.path,
                            "DATA_{}".format(modality_name.upper()),
                            "{}_{}.npy".format(modality_name.upper(), name),
                        ))), self.nb_split, part).to(torch.float32)
                dates = prepare_dates(line['-'.join(['dates', modality_name.upper()])], self.reference_date)
                l = []
                for i in range (4):
                    mask = ((dates >= 92 * i) & (dates < 92 * (i+1)))
                    if sum(mask) > 0:
                        r, _ = torch.median(images[mask], dim = 0)
                        l.append(r)
                    else:
                        l.append(torch.zeros((images.shape[1], images.shape[-2], images.shape[-1])))
                output[modality] = torch.cat(l)
            elif modality == "s2-4season-median":
                modality_name = "s2"
                images = split_image(torch.from_numpy(np.load(os.path.join(
                            self.path,
                            "DATA_{}".format(modality_name.upper()),
                            "{}_{}.npy".format(modality_name.upper(), name),
                        ))), self.nb_split, part).to(torch.float32)
                dates = prepare_dates(line['-'.join(['dates', modality_name.upper()])], self.reference_date)
                l = []
                for i in range (4):
                    mask = ((dates >= 92 * i) & (dates < 92 * (i+1)))
                    if sum(mask) > 0:
                        r, _ = torch.median(images[mask], dim = 0)
                        l.append(r)
                    else:
                        l.append(torch.zeros((images.shape[1], images.shape[-2], images.shape[-1])))
                output[modality] = torch.cat(l)
            elif modality == "s1":
                modality_name = "s1a"
                output[modality] = split_image(torch.from_numpy(np.load(os.path.join(
                            self.path,
                            "DATA_{}".format(modality_name.upper()),
                            "{}_{}.npy".format(modality_name.upper(), name),
                        ))), self.nb_split, part)
                output['_'.join([modality, 'dates'])] = prepare_dates(line['-'.join(['dates', modality_name.upper()])], self.reference_date)
                N = len(output[modality])
                if self.split == "train" and N > self.temporal_dropout:
                    random_indices = torch.randperm(N)[:self.temporal_dropout]
                    output[modality] = output[modality][random_indices]
                    output['_'.join([modality, 'dates'])] = output['_'.join([modality, 'dates'])][random_indices]
            elif modality == "s2-multi":
                modality_name = "s2"
                images = split_image(torch.from_numpy(np.load(os.path.join(
                            self.path,
                            "DATA_{}".format(modality_name.upper()),
                            "{}_{}.npy".format(modality_name.upper(), name),
                        ))), self.nb_split, part).to(torch.float32)
                output[modality] = images
                if self.super_res:
                    output[modality] = output[modality][int(temporal_split[0]):int(temporal_split[-1]), [2,1,0,6], ...]
            else:
                if len(modality) > 3:
                    modality_name = modality[:2] + modality[3]
                    output[modality] = split_image(torch.from_numpy(np.load(os.path.join(
                            self.path,
                            "DATA_{}".format(modality_name.upper()),
                            "{}_{}.npy".format(modality_name.upper(), name),
                        ))), self.nb_split, part)
                    output['_'.join([modality, 'dates'])] = prepare_dates(line['-'.join(['dates', modality_name.upper()])], self.reference_date)
                else:
                    output[modality] = split_image(torch.from_numpy(np.load(os.path.join(
                            self.path,
                            "DATA_{}".format(modality.upper()),
                            "{}_{}.npy".format(modality.upper(), name),
                        ))), self.nb_split, part)
                    output['_'.join([modality, 'dates'])] = prepare_dates(line['-'.join(['dates', modality.upper()])], self.reference_date)
                N = len(output[modality])
                if self.split == "train" and N > self.temporal_dropout:
                    random_indices = torch.randperm(N)[:self.temporal_dropout]
                    output[modality] = output[modality][random_indices]
                    output['_'.join([modality, 'dates'])] = output['_'.join([modality, 'dates'])][random_indices]
             
        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (output[modality] - self.norm[modality][0][None, :, 
                                                    None, None]) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (output[modality] - self.norm[modality][0][:, None, None]) / self.norm[modality][1][:, None, None]
        output = self.transform(output)
        hr = torch.cat([output['spot'][None,...], output['spot'][None,0:1,...]], dim=1)
        dict_return = {
            'img_hr': F.interpolate(hr, (512, 512), mode="bicubic").squeeze(),
            'img_lr': output['s2-multi'],
            'img_lr_up': F.interpolate(output['s2-multi'], (512, 512), mode='bicubic').squeeze(),
            'item_name': str(i),
            'indexes': i,
            'xy': str(i),
            'dates-S2': line['dates-S2'],
            'ID_PATCH': line['ID_PATCH'],
        }
        return dict_return

    def __len__(self):
        return len(self.meta_patch) * self.nb_split * self.nb_split