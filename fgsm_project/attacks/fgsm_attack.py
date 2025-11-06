import torch

# keep these in sync with your data loader
UNNORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
UNNORMALIZE_STD  = (0.2470, 0.2435, 0.2616)

def fgsm_attack(image, epsilon, data_grad):
    UNNORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
    UNNORMALIZE_STD  = (0.2470, 0.2435, 0.2616)

    sign_data_grad = data_grad.sign()
    std = torch.tensor(UNNORMALIZE_STD, device=image.device).view(1, 3, 1, 1)
    mean = torch.tensor(UNNORMALIZE_MEAN, device=image.device).view(1, 3, 1, 1)

    eps_norm = epsilon / std
    min_norm = (0 - mean) / std
    max_norm = (1 - mean) / std

    perturbed_image = image + eps_norm * sign_data_grad
    perturbed_image = torch.max(torch.min(perturbed_image, max_norm), min_norm)

    return perturbed_image
