import torch
import numpy as np

import lpips
#from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

import cv2 as cv


def minmax_normalize(tensor, eps=1e-7):
    """
    Apply min-max normalization to a batch of tensors across the channel dimension.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)
        eps (float): Small value to avoid division by zero in case the max equals min.
    
    Returns:
        torch.Tensor: Normalized tensor with the same shape as input.
    """
    # Compute the minimum and maximum values across the channel dimension
    min_val = tensor.amin(dim=(2, 3), keepdim=True)
    max_val = tensor.amax(dim=(2, 3), keepdim=True)
    
    # Normalize the tensor
    normalized_tensor = (tensor - min_val) / (max_val - min_val + eps)
    
    return normalized_tensor

class Measure:
    def __init__(self, net='alex'):
        self.model = lpips.LPIPS(net=net).to('cuda')
        #self.psnr = psnr(data_range=255.0, reduction="none").to('cuda')

    def measure(self, SR, HR, img_lr, sr_scale=4):
        """

        Args:
            SR: [B, C, H, W] uint8 or torch.FloatTensor [-1,1]
            HR: [B, C, H, W] uint8 or torch.FloatTensor [-1,1]
            img_lr: [B, C, H, W] uint8  or torch.FloatTensor [-1,1]
            sr_scale: 4

        Returns: dict of metrics

        """
        #import ipdb; ipdb.set_trace()
        if HR.shape[1]==3:
            lpips = self.lpips(SR, HR)
        else:
            lpips = self.lpips(SR[:,:3,...], HR[:,:3,...])
        
        
        if isinstance(SR, torch.Tensor):
            SR = np.clip(((SR.cpu().numpy()+1)/2 * 255), 0, 255)
            HR = np.clip(((HR.cpu().numpy()+1)/2 * 255), 0, 255)
        
        #SR = SR.transpose(0, 2, 3, 1)
        #HR = HR.transpose(0, 2, 3, 1)
        #img_lr = img_lr.transpose(1, 2, 0)
        psnr = self.psnr(SR, HR)
        ssim = self.ssim(SR, HR)
        #lr_psnr = self.psnr(SR_lr, img_lr)
        mae = self.mae(SR, HR)
        mse = self.mse(SR, HR)       
        shift_mae = self.shift_l1_loss(SR, HR)
        res = {'psnr': psnr,'ssim': ssim, 'lpips': lpips, 'mae': mae, 'mse': mse, "shift_mae": shift_mae}
        return res #{k: float(v) for k, v in res.items()}

    def lpips(self, SR, HR, model=None):
        #tA = SR.transpose().to(device)
        #tB = HR.transpose().to(device)
        SR = torch.clip(SR, -1.0, 1.0)
        dist01 = self.model.forward(SR, HR)
        return dist01.squeeze().cpu().numpy()

    def psnr(self, SR, HR, data_range=255.0):
        mse = ((SR - HR)**2).mean(axis=(1, 2, 3))
        psnr_base_e =  2 * np.log(data_range * np.ones(HR.shape[0])) - np.log(mse)
        return (psnr_base_e * (10 / np.log(10)))
    
    def ssim(self, SR, HR, data_range=255.0):
        res = []
        for b in range(HR.shape[0]):
            res.append(ssim_metric(SR[0], HR[0], channel_axis=0, multichannel=True, data_range=255.0))
        return np.array(res)
    
    def mae(self, SR, HR):
        return np.abs(SR - HR).mean(axis=(1, 2, 3))

    def mse(self, SR, HR):
        return ((SR - HR)**2).mean(axis=(1, 2, 3))

    def shift_l1_loss(self, SR, HR, border=3):
        """
        Modified mae to take into account pixel shifts
        """
        y_true = HR
        y_pred = SR
        max_pixels_shifts = 2*border
        size_image = y_true.shape[2]
        size_cropped_image = size_image - max_pixels_shifts
        patch_pred = y_pred[..., border:size_image -
                                    border, border:size_image-border]
        X = []
        for i in range(max_pixels_shifts+1):
            for j in range(max_pixels_shifts+1):
                patch_true = y_true[..., i:i+(size_image-max_pixels_shifts),
                                        j:j+(size_image-max_pixels_shifts)]
                l1_loss =np.abs(patch_true-patch_pred).mean()
                X.append(l1_loss)

        min_l1 = min(X)

        return min_l1