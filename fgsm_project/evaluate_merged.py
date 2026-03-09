# DF-RAP + FGSM Merged

# import os
# import torch
# import torch.nn.functional as F
# from models.cnn_model import SimpleCNN
# from models.gen_models import UNetGen
# from attacks.fgsm_attack import fgsm_attack
# from attacks.dfrap_attack import derive_df_rap
# from utils.dataset_loader import get_data_loaders
# from utils.metrics import JpegProxy, compute_psnr, compute_ssim
# from torchvision.utils import save_image

# # Normalization constants (Must match your dataset_loader.py)
# MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to('cuda' if torch.cuda.is_available() else 'cpu')
# STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to('cuda' if torch.cuda.is_available() else 'cpu')

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     os.makedirs("outputs/results", exist_ok=True)

#     # 1. Load the Target Classifier
#     model = SimpleCNN().to(device)
#     model.load_state_dict(torch.load("cnn_cifar10.pth", map_location=device))
#     model.eval()

#     # 2. Load the DF-RAP Generator you just trained
#     comgen = UNetGen().to(device)
#     comgen.load_state_dict(torch.load("models/comgen.pt", map_location=device))
#     comgen.eval()
    
#     jpeg_proxy = JpegProxy().to(device)
    
#     # 3. Load Test Images
#     test_loader, _ = get_data_loaders(batch_size=1, image_dir="./data/testData")

#     print(f"Starting Merged Pipeline on {device}...")

#     for i, (data, target) in enumerate(test_loader):
#         data, target = data.to(device), target.to(device)
        
#         # --- STAGE 1: DF-RAP PROTECTION ---
#         # Unnormalize to [0, 1] for processing
#         clean_01 = torch.clamp(data * STD + MEAN, 0, 1)
        
#         # Generate protection perturbation
#         eta = derive_df_rap(model, comgen, jpeg_proxy, clean_01, device=device, steps=50)
#         dfrap_img_01 = torch.clamp(clean_01 + eta, 0, 1)
        
#         # --- STAGE 2: FGSM ATTACK ---
#         # Re-normalize for the classifier
#         dfrap_norm = (dfrap_img_01 - MEAN) / STD
#         dfrap_norm.requires_grad = True
        
#         output = model(dfrap_norm)
#         fake_target = (output.argmax(dim=1) + 1) % output.size(1)
#         loss = F.cross_entropy(output, fake_target)
        
#         model.zero_grad()
#         loss.backward()
        
#         # Add FGSM noise (Epsilon = 0.03)
#         final_adv_norm = fgsm_attack(dfrap_norm, 0.03, dfrap_norm.grad.data)
#         final_img_01 = torch.clamp(final_adv_norm * STD + MEAN, 0, 1)
        
#         # --- STAGE 3: EVALUATION ---
#         psnr_score = compute_psnr(clean_01, final_img_01)
#         ssim_score = compute_ssim(clean_01, final_img_01)
        
#         print(f"Image {i} | PSNR: {psnr_score:.2f} dB | SSIM: {ssim_score:.4f}")
        
#         # Save output
#         save_image(final_img_01, f"outputs/results/final_adversarial_{i}.png")

#         if i >= 9: break # Run for first 10 images

#     print("\nPipeline complete. Check 'outputs/results' for the images.")

# if __name__ == "__main__":
#     main()


# FGSM + DF-RAP Merged

import os
import torch
import torch.nn.functional as F
from models.cnn_model import SimpleCNN
from models.gen_models import UNetGen
from attacks.fgsm_attack import fgsm_attack
from attacks.dfrap_attack import derive_df_rap
from utils.dataset_loader import get_data_loaders
from utils.metrics import JpegProxy, compute_psnr, compute_ssim
from torchvision.utils import save_image

# 1. Device and Normalization Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Normalization constants must match your dataset_loader.py (CIFAR-10 stats)
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(DEVICE)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(DEVICE)

def main():
    os.makedirs("outputs/results", exist_ok=True)

    # 2. Load the Classifier (Target Model)
    model = SimpleCNN().to(DEVICE)
    if os.path.exists("cnn_cifar10.pth"):
        model.load_state_dict(torch.load("cnn_cifar10.pth", map_location=DEVICE))
    model.eval()

    # 3. Load the DF-RAP Generator (Trained ComGen)
    comgen = UNetGen().to(DEVICE)
    if os.path.exists("models/comgen.pt"):
        comgen.load_state_dict(torch.load("models/comgen.pt", map_location=DEVICE))
    comgen.eval()
    
    jpeg_proxy = JpegProxy().to(DEVICE)
    
    # 4. Load Test Dataset
    test_loader, _ = get_data_loaders(batch_size=1, image_dir="./data/testData")

    print(f"Starting Fixed Reversed Pipeline (FGSM -> DF-RAP) on {DEVICE}...")

    for i, (data, target) in enumerate(test_loader):
        data = data.to(DEVICE)
        
        # --- STAGE 1: FGSM ATTACK ---
        # We need gradients on the original image for FGSM
        data.requires_grad = True
        output = model(data)
        
        # Untargeted attack: move away from current prediction
        fake_target = (output.argmax(dim=1) + 1) % output.size(1)
        loss_fgsm = F.cross_entropy(output, fake_target)
        
        model.zero_grad()
        loss_fgsm.backward()
        
        # Generate FGSM perturbed image (normalized range)
        # eps=0.03 is standard for subtle adversarial noise
        fgsm_img_norm = fgsm_attack(data, 0.03, data.grad.data)
        
        # --- STAGE 2: DF-RAP PROTECTION ---
        # !!! CRITICAL FIX !!! 
        # .detach() removes the FGSM gradient history so DF-RAP can start a new graph
        fgsm_img_01 = torch.clamp(fgsm_img_norm.detach() * STD + MEAN, 0, 1)
        
        # Run DF-RAP optimization loop on the FGSM output
        # derive_df_rap uses internal backward() calls to optimize eta
        eta = derive_df_rap(model, comgen, jpeg_proxy, fgsm_img_01, device=DEVICE, steps=50)
        
        # Final combined adversarial image [0, 1]
        final_img_01 = torch.clamp(fgsm_img_01 + eta, 0, 1)
        
        # --- STAGE 3: EVALUATION & SAVING ---
        clean_01 = torch.clamp(data.detach() * STD + MEAN, 0, 1)
        
        psnr_score = compute_psnr(clean_01, final_img_01)
        ssim_score = compute_ssim(clean_01, final_img_01)
        
        print(f"Image {i} | PSNR: {psnr_score:.2f} dB | SSIM: {ssim_score:.4f}")
        
        # Save as lossless PNG to preserve perturbation integrity
        save_image(final_img_01, f"outputs/results/reversed_adv_{i}.png")

        if i >= 9: # Process first 10 images for the report
            break

    print("\nPipeline complete. Results saved in 'outputs/results/'.")

if __name__ == "__main__":
    main()