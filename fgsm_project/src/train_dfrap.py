import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Allows script to see folders outside of 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.gen_models import UNetGen, PatchDiscriminator
from utils.metrics import JpegProxy
from utils.dataset_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_comgan(image_dir, epochs=5):
    # Use your existing data loader
    loader, _ = get_data_loaders(batch_size=8, image_dir=image_dir)
    
    gen = UNetGen().to(device)
    disc = PatchDiscriminator().to(device)
    jpeg_proxy = JpegProxy().to(device)

    opt_g = optim.Adam(gen.parameters(), lr=1e-4)
    opt_d = optim.Adam(disc.parameters(), lr=1e-4)
    l1 = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()

    print(f"Starting GAN Training for DF-RAP on {device}...")

    for ep in range(epochs):
        for imgs, _ in tqdm(loader, desc=f"Epoch {ep+1}/{epochs}"):
            imgs = imgs.to(device)
            
            # 1. Train Discriminator
            real_comp = jpeg_proxy(imgs)
            fake = gen(imgs)
            fake_comp = jpeg_proxy(fake)

            d_real = disc(real_comp)
            d_fake = disc(fake_comp.detach())
            loss_d = 0.5 * (bce(d_real, torch.ones_like(d_real)) +
                            bce(d_fake, torch.zeros_like(d_fake)))
            
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # 2. Train Generator
            d_fake_pred = disc(fake_comp)
            loss_g_adv = bce(d_fake_pred, torch.ones_like(d_fake_pred))
            loss_rec = l1(fake_comp, real_comp)
            loss_g = loss_g_adv + 10 * loss_rec

            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

    os.makedirs("models", exist_ok=True)
    torch.save(gen.state_dict(), "models/comgen.pt")
    return gen

def start_training():
    print("Initializing Training Process...")
    dataset_folder = "./data/trainData" 
    
    if not os.path.exists(dataset_folder) or not os.listdir(dataset_folder):
        print(f"Error: No images found in {dataset_folder}")
        return

    train_comgan(dataset_folder, epochs=5)
    print("Success! 'models/comgen.pt' has been saved.")

if __name__ == "__main__":
    start_training()