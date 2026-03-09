import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def derive_df_rap(target_model, comgen, jpeg_proxy, imgs, device='cuda',
                  eps=8/255, steps=200, lr=0.01):
    B, C, H, W = imgs.shape
    # Initialize perturbation
    eta = (torch.rand_like(imgs) - 0.5) * eps
    eta = eta.to(device)
    eta.requires_grad = True

    opt = optim.Adam([eta], lr=lr)

    for step in range(steps):
        opt.zero_grad()
        x_p = torch.clamp(imgs + eta, 0, 1)
        
        with torch.no_grad():
            cg_out = comgen(x_p)
            cg_out = jpeg_proxy(cg_out)
            
        M_orig = target_model(imgs)
        M_adv = target_model(cg_out)
        
        # We want to maximize the difference (MSE)
        loss = -nn.functional.mse_loss(M_orig, M_adv)
        loss.backward()
        opt.step()
        
        # Ensure perturbation stays within epsilon bounds
        eta.data = torch.clamp(eta.data, -eps, eps)

    return eta.detach()