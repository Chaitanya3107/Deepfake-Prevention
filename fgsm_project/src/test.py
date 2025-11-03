# src/test_fgsm.py
import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.cnn_model import SimpleCNN
from utils.dataset_loader import get_data_loaders
from attacks.fgsm_attack import fgsm_attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change these if you used different values in dataset loader
UNNORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
UNNORMALIZE_STD  = (0.2470, 0.2435, 0.2616)

def unnormalize_tensor(tensor):
    t = tensor.clone().cpu()
    for c in range(3):
        t[c] = t[c] * UNNORMALIZE_STD[c] + UNNORMALIZE_MEAN[c]
    return torch.clamp(t, 0.0, 1.0)

def evaluate(epsilon=0.03, batch_save=3):
    # Load data
    test_loader, _ = get_data_loaders()  # or if your loader returns two
    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("cnn_cifar10.pth", map_location=device))
    model.eval()

    # Set up output folder
    out_root = f"outputs/images_test/eps_{epsilon:.4f}"
    os.makedirs(out_root, exist_ok=True)

    correct = 0
    total = 0
    psnr_vals = []
    ssim_vals = []

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # Forward pass clean
        output = model(data)
        init_pred = output.argmax(dim=1)

        # Compute loss and gradient for FGSM
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # Craft adversarial
        adv_data = fgsm_attack(data, epsilon, data_grad)

        # Evaluate
        output_adv = model(adv_data)
        pred_adv = output_adv.argmax(dim=1)
        correct += (pred_adv == target).sum().item()
        total   += target.size(0)

        # Save some images
        if batch_idx < batch_save:
            nsave = min(data.size(0), 5)
            for i in range(nsave):
                clean_img = unnormalize_tensor(data[i].detach())
                adv_img   = unnormalize_tensor(adv_data[i].detach())

                clean_path = os.path.join(out_root, f"batch{batch_idx}_img{i}_clean.jpg")
                adv_path   = os.path.join(out_root, f"batch{batch_idx}_img{i}_adv.jpg")

                save_image(clean_img, clean_path)
                save_image(adv_img, adv_path)

        # Compute PSNR/SSIM on maybe first image of the batch
        orig_np = unnormalize_tensor(data[0].detach()).permute(1,2,0).numpy()
        adv_np  = unnormalize_tensor(adv_data[0].detach()).permute(1,2,0).numpy()
        psnr_vals.append(peak_signal_noise_ratio(orig_np, adv_np, data_range=1.0))
        ssim_vals.append(structural_similarity(orig_np, adv_np, channel_axis=2, data_range=1.0))

    acc = 100.0 * correct / total
    mean_psnr = float(np.mean(psnr_vals))
    mean_ssim = float(np.mean(ssim_vals))

    print(f"Epsilon: {epsilon:.4f} | Adversarial Accuracy: {acc:.2f}% | Mean PSNR: {mean_psnr:.2f} dB | Mean SSIM: {mean_ssim:.4f}")

if __name__ == "__main__":
    for eps in [0.01, 0.02, 0.03, 0.05]:
        evaluate(epsilon=eps)
