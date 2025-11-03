import torch
import torch.nn.functional as F
from models.cnn_model import SimpleCNN
from utils.dataset_loader import get_data_loaders
from attacks.fgsm_attack import fgsm_attack
from utils.metrics import compute_psnr, compute_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(epsilon):
    _, test_loader = get_data_loaders()
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("cnn_mnist.pth"))
    model.eval()

    correct = 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        correct += final_pred.eq(target.view_as(final_pred)).sum().item()

        psnr = compute_psnr(data[0], perturbed_data[0])
        ssim = compute_ssim(data[0], perturbed_data[0])

        print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
        break  # only run one batch for test

    print(f"Accuracy after FGSM attack (ε={epsilon}): {correct / len(test_loader.dataset) * 100:.2f}%")

if __name__ == "__main__":
    test(0.2)
