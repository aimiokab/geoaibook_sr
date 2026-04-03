import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from skimage import exposure


def stretch_standardize(img, type, mode="sisr", nb_channels=3):
    if type == "spot6":
        if nb_channels==3:
            transform = transforms.Compose([transforms.Normalize(mean=[0.5 for i in range(nb_channels)], std=[0.5 for i in range(nb_channels)])])
            return transform(img)
        elif nb_channels==4:
            if mode=="sisr":
                return stretch_standardize_utils(img, type="spot6", nb_channels=nb_channels)
            else :
                for e in range(img.shape[0]):
                    img[e] = stretch_standardize_utils(img[e], type="spot6", nb_channels=nb_channels)
                return img
    elif type == "sen2":
        if mode=="sisr":
            return stretch_standardize_utils(img, type="sen2", nb_channels=nb_channels)
        else :
            for e in range(img.shape[0]):
                img[e] = stretch_standardize_utils(img[e], type="sen2", nb_channels=nb_channels)
            return img


def stretch_standardize_utils(img, type="sen2", nb_channels=3):
    if nb_channels == 3 and type=="sen2":
        min = np.array([0,0,0]) #np.array([0.0027,0.0139,0.0040,0.0086])
        max = np.array([0.2058, 0.1583, 0.1118]) #np.array([0.1699, 0.1388, 0.0997, 0.5449]) #98th percentile over whole train set
    if nb_channels == 4:
        min = np.array([0,0,0,0])
        if type=="sen2":
            max = np.array([0.1970, 0.1487, 0.1072, 0.5223])
        else:
            max = np.array([0.199267, 0.209768, 0.203907, 0.531136])

    for i in range(nb_channels):
        img[i] = img[i].clamp(min[i], max[i])
    transform = transforms.Compose([transforms.Normalize(mean=min, std=max-min)])
    return 2*transform(img).float()-1


"""
SPECTRAL MATCHING METHODS
"""
# Normal Standardization over whole dataset
def normalize(sen2,spot6,sen2_amount=1):
    transform_spot = transforms.Compose([transforms.Normalize(mean=[479.0, 537.0, 344.0], std=[430.0, 290.0, 229.0]) ])
    # dynamically define transform to reflect shape of tensor
    trans_mean,trans_std = [78.0, 91.0, 62.0]*sen2_amount,[36.0, 28.0, 30.0]*sen2_amount
    transform_sen = transforms.Compose([transforms.Normalize(mean=trans_mean, std= trans_std)])
    # perform transform
    sen2  = transform_sen(sen2)
    spot6 = transform_spot(spot6)
    return sen2,spot6

# HISTOGRAM MATCHING
def histogram(sen2,spot6,sen2_amount=None):
    sen2,spot6 = sen2.numpy(),spot6.numpy()
    sen2 = np.transpose(sen2,(1,2,0))
    spot6 = np.transpose(spot6,(1,2,0))
    spot6 = exposure.match_histograms(image=spot6,reference=sen2,channel_axis=2)
    spot6,sen2 = np.transpose(spot6,(2,0,1)),np.transpose(sen2,(2,0,1))
    spot6,sen2 = torch.Tensor(spot6),torch.Tensor(sen2)
    return spot6

# MOMENT MATCHING
def moment(sen2,spot6,sen2_amount=None):   
    sen2,spot6 = sen2.numpy(),spot6.numpy()
    c = 0
    for channel_sen,channel_spot in zip(sen2,spot6):
        c +=1
        #calculate stats
        sen2_mean   = np.mean(channel_sen)
        spot6_mean  = np.mean(channel_spot)
        sen2_stdev  = np.std(channel_sen)
        spot6_stdev = np.std(channel_spot)

        # calculate moment per channel
        channel_result = (((channel_spot - spot6_mean) / spot6_stdev) * sen2_stdev) + sen2_mean

        # stack channels to single array
        if c==1:
            spot6 = channel_result
        else:
            spot6 = np.dstack((spot6,channel_result))
        # transpose back to Cx..

    spot6 = torch.Tensor(spot6.transpose((2,0,1)))   
    return spot6 


