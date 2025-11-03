# src/train.py
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

from models.cnn_model import SimpleCNN
from utils.dataset_loader import get_data_loaders
from attacks.fgsm_attack import fgsm_attack   # your fgsm function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR mean/std used in dataset transforms (change if you used different values)
UNNORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
UNNORMALIZE_STD  = (0.2470, 0.2435, 0.2616)

def unnormalize_tensor(tensor):
    """Convert a normalized tensor (C,H,W) -> tensor in [0,1] (C,H,W)."""
    t = tensor.clone().cpu()
    for c in range(3):
        t[c] = t[c] * UNNORMALIZE_STD[c] + UNNORMALIZE_MEAN[c]
    return torch.clamp(t, 0.0, 1.0)

def train(num_epochs=3, epsilon=0.03):
    train_loader, _ , = get_data_loaders()  # keep old signature compatibility
    # ensure outputs root exists
    base_out = "outputs/images"
    os.makedirs(base_out, exist_ok=True)
    counter_file = os.path.join(base_out, "run_counter.txt")

    # initialize counter if not exists
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as f:
            f.write("0")

    # read current run number
    with open(counter_file, "r") as f:
        run_num = int(f.read().strip())

    # increment and save it back
    run_num += 1
    with open(counter_file, "w") as f:
        f.write(str(run_num))

    # define current run folder
    run_folder = os.path.join(base_out, f"run_{run_num:04d}")
    os.makedirs(run_folder, exist_ok=True)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_dir = os.path.join(run_folder, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # we need gradients w.r.t. input to craft FGSM
            data.requires_grad = True

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            # compute gradients w.r.t. model params & input
            loss.backward()

            # grab gradient on the input and craft adversarial examples
            data_grad = data.grad.data
            adv_data = fgsm_attack(data, epsilon, data_grad)  # returns clamped tensor

            # optimizer step updates model parameters (trained on clean input here)
            optimizer.step()

            # Save a few examples (clean + adversarial) from early batches
            if batch_idx < 2:
                nsave = min(8, data.size(0))
                for i in range(nsave):
                    clean_unn = unnormalize_tensor(data[i].detach())
                    adv_unn   = unnormalize_tensor(adv_data[i].detach())

                    clean_path = os.path.join(epoch_dir, f"batch{batch_idx}_img{i}_clean.jpg")
                    adv_path   = os.path.join(epoch_dir, f"batch{batch_idx}_img{i}_adv.jpg")

                    save_image(clean_unn, clean_path)
                    save_image(adv_unn, adv_path)

        print(f"✅ Epoch {epoch} completed — saved samples to {epoch_dir}")

    # final save
    model_path = "cnn_cifar10.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved as {model_path}")


if __name__ == "__main__":
    # tweak epochs and epsilon here if you want
    train(num_epochs=2, epsilon=0.01)
