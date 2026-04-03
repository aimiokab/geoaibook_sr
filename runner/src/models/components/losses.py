import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg19
import torchvision.transforms as transforms


def sobel_filter():
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0).cuda()
    
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [0.,  0.,  0.],
                            [1.,  2.,  1.]]).unsqueeze(0).unsqueeze(0).cuda()
    
    return sobel_x, sobel_y

def compute_gradients(image):
    sobel_x, sobel_y = sobel_filter()
    grad_x = F.conv2d(image, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    grad_y = F.conv2d(image, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
    return grad_x, grad_y

def gradient_loss(sr_image, hr_image):
    sr_grad_x, sr_grad_y = compute_gradients(sr_image)
    hr_grad_x, hr_grad_y = compute_gradients(hr_image)
    
    loss_x = F.l1_loss(sr_grad_x, hr_grad_x)
    loss_y = F.l1_loss(sr_grad_y, hr_grad_y)
    
    return loss_x + loss_y


# VGG loss 

def gram_matrix(input_tensor):
    (b, c, h, w) = input_tensor.size()
    features = input_tensor.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)

class StyleLossVGG(nn.Module):
    def __init__(self, model):
        super(StyleLossVGG, self).__init__()
        self.model = model

    def forward(self, style_img, generated_img):
        style_activations = self.model(style_img * 255.0 / 12.75)
        generated_activations = self.model(generated_img * 255.0 / 12.75)
        loss = 0.0
        for style, gen in zip(style_activations, generated_activations):
            style_matrix = gram_matrix(style)
            gen_matrix = gram_matrix(gen)
            loss += F.mse_loss(style_matrix, gen_matrix)
        return loss / len(style_activations)


class VGGContentLoss(nn.Module):
    def __init__(self, criterion='l2', layers=[8, 35]):
        super(VGGContentLoss, self).__init__()
        self.model = vgg19(pretrained=True).features
        self.layers = layers
        self.preprocess = transforms.Compose([
                        transforms.Lambda(lambda x: (x + 1)/2),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        for param in self.model.parameters():
            param.requires_grad = False

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError(f"Loss type {criterion} not recognized.")

    def forward(self, sr, hr):
        loss = torch.zeros(hr.size(0), device=sr.device)
        x = self.preprocess(sr)
        y = self.preprocess(hr)
        layer = 0
        for layer_index in self.layers:
            x = self.model[layer:layer_index+1](x)
            y = self.model[layer:layer_index+1](y)
            loss += self.criterion(x/12.75, y/12.75).mean(dim=[1, 2, 3])
            layer = layer_index+1
        return loss


def temp_weighted_MSE_Loss(pred, ut, temporal_weights):
    mse = F.mse_loss(pred, ut, reduction='none').mean(dim=[1, 2, 3])
    weighted_mse = mse*temporal_weights
    return weighted_mse.sum()

