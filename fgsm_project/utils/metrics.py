# import torch
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# def compute_psnr(img1, img2):
#     img1_np = img1.squeeze().detach().cpu().numpy()
#     img2_np = img2.squeeze().detach().cpu().numpy()
#     return peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)

# def compute_ssim(img1, img2):
#     img1_np = img1.squeeze().detach().cpu().numpy()
#     img2_np = img2.squeeze().detach().cpu().numpy()
#     return structural_similarity(img1_np, img2_np, data_range=1.0)

import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class JpegProxy(nn.Module):
    def __init__(self, q_list=[70, 80, 90, 95]):
        super().__init__()
        self.q_list = q_list

    def forward(self, x):
        B, C, H, W = x.shape
        out_imgs = []
        # Randomly apply quality factors to simulate robust deepfake conditions
        for i in range(B):
            img = x[i:i+1].clone()
            scale = np.random.choice(self.q_list) / 100.0
            noise = (1 - scale) * (torch.rand_like(img) - 0.5) * 0.06
            img = torch.clamp(img + noise, 0, 1)
            out_imgs.append(img)
        return torch.cat(out_imgs, dim=0)

def compute_psnr(img_clean, img_adv):
    # Expects [0,1] tensors
    c = img_clean.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    a = img_adv.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return peak_signal_noise_ratio(c, a, data_range=1.0)

def compute_ssim(img_clean, img_adv):
    c = img_clean.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    a = img_adv.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return structural_similarity(c, a, channel_axis=2, data_range=1.0)