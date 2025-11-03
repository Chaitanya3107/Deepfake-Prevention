import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(img1, img2):
    img1_np = img1.squeeze().detach().cpu().numpy()
    img2_np = img2.squeeze().detach().cpu().numpy()
    return peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)

def compute_ssim(img1, img2):
    img1_np = img1.squeeze().detach().cpu().numpy()
    img2_np = img2.squeeze().detach().cpu().numpy()
    return structural_similarity(img1_np, img2_np, data_range=1.0)
