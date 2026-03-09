# import os
# import torch
# import torch.nn.functional as F
# import numpy as np
# from torchvision.utils import save_image, make_grid
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# from models.cnn_model import SimpleCNN
# from utils.dataset_loader import get_data_loaders
# from attacks.fgsm_attack import fgsm_attack

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Normalization constants for CIFAR-like data
# UNNORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
# UNNORMALIZE_STD  = (0.2470, 0.2435, 0.2616)

# def unnormalize_tensor(tensor):
#     """
#     Convert normalized tensor back to its original image range [0,1].
#     """
#     t = tensor.clone().cpu()
#     for c in range(3):
#         t[c] = t[c] * UNNORMALIZE_STD[c] + UNNORMALIZE_MEAN[c]
#     return torch.clamp(t, 0.0, 1.0)


# def evaluate(epsilon=0.03, batch_save=3, verbose_debug=False):
#     """
#     Run FGSM adversarial attack evaluation.

#     Args:
#         epsilon: Magnitude of FGSM perturbation.
#         batch_save: Number of batches for which to save sample outputs.
#         verbose_debug: If True, prints detailed debug information.
#     """
#     # Load test data
#     test_loader, _ = get_data_loaders(image_dir="./data/testData")

#     # Load trained model
#     model = SimpleCNN().to(device)
#     model.load_state_dict(torch.load("cnn_cifar10.pth", map_location=device))
#     model.eval()

#     # Create output folder for this epsilon
#     out_root = f"outputs/images_test/eps_{epsilon:.4f}"
#     os.makedirs(out_root, exist_ok=True)

#     # Metrics
#     correct, total = 0, 0
#     psnr_vals, ssim_vals = [], []

#     # Check if model parameters require grad
#     if verbose_debug:
#         print("Model has trainable params:", any(p.requires_grad for p in model.parameters()))

#     # Loop over test batches
#     for batch_idx, (data, target) in enumerate(test_loader):
#         data, target = data.to(device), target.to(device)
#         target = target.long()   # ensure correct dtype for loss

#         with torch.set_grad_enabled(True):
#             data.requires_grad = True
#             model.zero_grad()  # clear gradients

#             # Forward pass
#             output = model(data)

#             # ⚠️ For unlabeled data: use fake targets to force nonzero loss
#             fake_target = (output.argmax(dim=1) + 1) % output.size(1)
#             loss = F.cross_entropy(output, fake_target)

#             if verbose_debug:
#                 print(f"Batch {batch_idx} - loss: {loss.item():.6f}")

#             # Backward pass to compute gradients w.r.t. input
#             loss.backward()

#             # Extract gradient
#             data_grad = data.grad.data if data.grad is not None else torch.zeros_like(data)

#             # Generate adversarial example using FGSM
#             adv_data = fgsm_attack(data, epsilon, data_grad)

#             # Calculate perturbation magnitude (for info)
#             pert_norm = (adv_data - data).abs().mean().item()
#             if verbose_debug:
#                 print(f"Batch {batch_idx} - mean perturbation: {pert_norm:.6e}")

#         # Evaluate adversarial prediction
#         with torch.no_grad():
#             output_adv = model(adv_data)
#             pred_adv = output_adv.argmax(dim=1)
#             correct += (pred_adv == target).sum().item()
#             total += target.size(0)

#         # -------------------------------
#         # SAVE SAMPLE IMAGES AND VISUALIZATIONS
#         # -------------------------------
#             nsave = min(data.size(0), 5)

#             # Save a few clean and adversarial images
#             for i in range(nsave):
#                 clean_img = unnormalize_tensor(data[i].detach())
#                 adv_img   = unnormalize_tensor(adv_data[i].detach())

#                 clean_path = os.path.join(out_root, f"batch{batch_idx}_img{i}_clean.jpg")
#                 adv_path   = os.path.join(out_root, f"batch{batch_idx}_img{i}_adv.jpg")

#                 save_image(clean_img, clean_path)
#                 save_image(adv_img, adv_path)

#             # 🔹 Save side-by-side comparison grid (only for first batch)
#             if batch_idx == 0:
#                 comparison = torch.cat([
#                     data[:5].detach().cpu(),
#                     adv_data[:5].detach().cpu()
#                 ])
#                 grid = make_grid(unnormalize_tensor(comparison), nrow=5)
#                 grid_path = os.path.join(out_root, f"comparison_eps{epsilon:.05f}.jpg")
#                 save_image(grid, grid_path)
#                 # print(f"🖼️ Saved side-by-side comparison to {grid_path}")

#                 # (Optional) Save perturbation heatmap visualization
#                 diff = (adv_data - data).abs()
#                 diff_grid = make_grid(unnormalize_tensor(diff * 10), nrow=5)
#                 diff_path = os.path.join(out_root, f"perturbation_eps{epsilon:.2f}.jpg")
#                 save_image(diff_grid, diff_path)
#                 # print(f"🌀 Saved perturbation visualization to {diff_path}")

#         # Compute PSNR and SSIM metrics
#         orig_np = unnormalize_tensor(data[0].detach()).permute(1, 2, 0).numpy()
#         adv_np  = unnormalize_tensor(adv_data[0].detach()).permute(1, 2, 0).numpy()
#         psnr_vals.append(peak_signal_noise_ratio(orig_np, adv_np, data_range=1.0))
#         ssim_vals.append(structural_similarity(orig_np, adv_np, channel_axis=2, data_range=1.0))

#     # Aggregate results
#     acc = 98.0 * correct / total if total > 0 else 0
#     mean_psnr = float(np.mean(psnr_vals))
#     mean_ssim = float(np.mean(ssim_vals))

#     print(f"\nEpsilon: {epsilon:.4f}"
#           f"Mean PSNR: {mean_psnr:.2f} dB | Mean SSIM: {mean_ssim:.4f}")


# if __name__ == "__main__":
#     # Run evaluation for multiple epsilon values
#     for eps in [0.01, 0.02, 0.03, 0.05,0.1]:
#         evaluate(epsilon=eps)

import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image, make_grid
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
from torchvision.transforms.functional import resize as tv_resize

from models.cnn_model import SimpleCNN
from utils.dataset_loader import get_data_loaders
from attacks.fgsm_attack import fgsm_attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNNORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
UNNORMALIZE_STD  = (0.2470, 0.2435, 0.2616)

def unnormalize_tensor(tensor):
    t = tensor.clone().cpu()
    for c in range(3):
        t[c] = t[c] * UNNORMALIZE_STD[c] + UNNORMALIZE_MEAN[c]
    return torch.clamp(t, 0.0, 1.0)


def evaluate(epsilon=0.03):
    test_loader, _ = get_data_loaders(image_dir="./data/testData")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("cnn_cifar10.pth", map_location=device))
    model.eval()

    out_root = f"outputs/images_test/eps_{epsilon:.4f}"
    os.makedirs(out_root, exist_ok=True)

    psnr_vals, ssim_vals = [], []

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.long().to(device)

        data.requires_grad = True
        model.zero_grad()

        output = model(data)
        fake_target = (output.argmax(dim=1) + 1) % output.size(1)
        loss = F.cross_entropy(output, fake_target)
        loss.backward()

        grad = data.grad.data
        adv_data = fgsm_attack(data, epsilon, grad)

        batch_size = data.size(0)

        # Loop over each image in batch
        for i in range(batch_size):
            clean_img = unnormalize_tensor(data[i])
            adv_img   = unnormalize_tensor(adv_data[i])

            # Original filename
            orig_filename = test_loader.dataset.image_files[batch_idx * batch_size + i]
            orig_path = os.path.join(test_loader.dataset.image_dir, orig_filename)

            orig_pil = Image.open(orig_path).convert("RGB")
            W, H = orig_pil.size

            # Resize back to original
            clean_resized = tv_resize(clean_img, (H, W))
            adv_resized   = tv_resize(adv_img, (H, W))

            # Make image folder
            img_folder = os.path.join(out_root, orig_filename.split('.')[0])
            os.makedirs(img_folder, exist_ok=True)

            # Paths
            clean_path = os.path.join(img_folder, "clean.jpg")
            adv_path   = os.path.join(img_folder, "adv.jpg")
            diff_path  = os.path.join(img_folder, "diff.jpg")
            comp_path  = os.path.join(img_folder, "comparison.jpg")

            # Save clean + adv
            save_image(clean_resized, clean_path)
            save_image(adv_resized, adv_path)

            # Diff (perturbation heatmap)
            diff_tensor = (adv_resized - clean_resized).abs() * 10
            save_image(diff_tensor, diff_path)

            # Comparison side-by-side
            comparison = torch.cat([clean_resized, adv_resized], dim=2)
            save_image(comparison, comp_path)

            # Metrics
            np_clean = clean_resized.detach().permute(1,2,0).cpu().numpy()
            np_adv   = adv_resized.detach().permute(1,2,0).cpu().numpy()

            psnr_vals.append(peak_signal_noise_ratio(np_clean, np_adv, data_range=1.0))
            ssim_vals.append(structural_similarity(np_clean, np_adv, channel_axis=2, data_range=1.0))

        print(f"Epsilon: {epsilon:.4f} | PSNR: {np.mean(psnr_vals):.2f} dB | SSIM: {np.mean(ssim_vals):.4f}")

if __name__ == "__main__":
    for eps in [0.01, 0.02, 0.03, 0.05, 0.1]:
        evaluate(epsilon=eps)
