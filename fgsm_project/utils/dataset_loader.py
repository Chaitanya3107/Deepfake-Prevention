# src/utils/dataset_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

IMG_SIZE = 224   # <- change here to desired human-viewable size

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = 0
        return image, target

def get_data_loaders(batch_size=3):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    dataset = CustomImageDataset(image_dir="./data", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, loader
