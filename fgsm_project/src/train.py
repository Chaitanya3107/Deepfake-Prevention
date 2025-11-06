import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

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


def train(num_epochs=3, epsilon=0.03):
    train_loader, _ = get_data_loaders(image_dir="./data/trainData")
    subset = list(train_loader)[:1500 // train_loader.batch_size]
    train_loader = subset
    base_out = "outputs/images"
    os.makedirs(base_out, exist_ok=True)
    counter_file = os.path.join(base_out, "run_counter.txt")

    if not os.path.exists(counter_file):
        with open(counter_file, "w") as f:
            f.write("0")

    with open(counter_file, "r") as f:
        run_num = int(f.read().strip())

    run_num += 1
    with open(counter_file, "w") as f:
        f.write(str(run_num))

    run_folder = os.path.join(base_out, f"run_{run_num:04d}")
    os.makedirs(run_folder, exist_ok=True)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_dir = os.path.join(run_folder, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        count = 1
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Image {count} training...")
            data, target = data.to(device), target.to(device)
            data.requires_grad = True

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            data_grad = data.grad.data
            adv_data = fgsm_attack(data, epsilon, data_grad)
            optimizer.step()

            if batch_idx < 2:
                nsave = min(8, data.size(0))
                for i in range(nsave):
                    clean_unn = unnormalize_tensor(data[i].detach())
                    adv_unn   = unnormalize_tensor(adv_data[i].detach())

                    clean_path = os.path.join(epoch_dir, f"batch{batch_idx}_img{i}_clean.jpg")
                    adv_path   = os.path.join(epoch_dir, f"batch{batch_idx}_img{i}_adv.jpg")

                    save_image(clean_unn, clean_path)
                    save_image(adv_unn, adv_path)
            count+=1

        print(f"✅ Epoch {epoch} completed — saved samples to {epoch_dir}")

    model_path = "cnn_cifar10.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved as {model_path}")


if __name__ == "__main__":
    train(num_epochs=2, epsilon=0.01)
