import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import torch


# def tensor2img(tensor, out_type=np.uint16, min_max=(-1, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     tensor = tensor.squeeze().float().cpu() #.clamp_(*min_max)  # clamp
#     tensor = (tensor - min_max[0]) / \
#         (min_max[1] - min_max[0])  # to range [0,1]
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         tensor
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(
#             math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError(
#             'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
#     if out_type == np.uint16:
#         img_np = (img_np * 65535.0).round()
#     return img_np.astype(out_type)

def tensor2img(tensor, out_type=np.uint16, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu()  # Remove extra dimensions and move to CPU

    # Handle 4D tensors (batch of images)
    if tensor.dim() == 4:
        # Normalize each image in the batch independently
        tensor = torch.stack([(img - img.min()) / (img.max() - img.min() + 1e-8) for img in tensor])
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB

    # Handle 3D tensors (single image with channels)
    elif tensor.dim() == 3:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)  # Normalize to [0, 1]
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB

    # Handle 2D tensors (grayscale image)
    elif tensor.dim() == 2:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)  # Normalize to [0, 1]
        img_np = tensor.numpy()

    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(tensor.dim())
        )

    # Convert to the desired output type
    if out_type == np.uint16:
        img_np = (img_np * 65535.0).round()

    return img_np.astype(out_type)

def save_img(img, img_path, mode='RGB'):
    """
    Save a uint16 or float image as visible uint8 PNG.
    Normalizes based on actual image intensity range.
    """
    # Convert to float32 if not already
    img = img.astype(np.float32)

    # Normalize dynamically based on actual data range
    img_min, img_max = img.min(), img.max()

    if img_max == img_min:
        img = np.zeros_like(img, dtype=np.uint8)  # avoid div by 0
    else:
        img = (img - img_min) / (img_max - img_min) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    if mode == 'RGB' and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(img_path, img)



def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(65535.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 65535)**2
    C2 = (0.03 * 65535)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
