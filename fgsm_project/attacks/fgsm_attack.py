import torch

def fgsm_attack(image, epsilon, data_grad):
    # sign of gradient
    sign_data_grad = data_grad.sign()
    # create adversarial image
    perturbed_image = image + epsilon * sign_data_grad
    # clip values to valid range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
