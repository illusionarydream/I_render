from torch.utils.data import Dataset
from PIL import Image
import os


class ImagePairDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        file_path 文件中，每一行是：<noisy_image_path> <clean_image_path>
        """
        with open(file_path, "r") as f:
            self.image_pairs = [line.strip().split(",") for line in f.readlines()]
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.image_pairs[idx]
        noisy = Image.open(noisy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
        return noisy, clean
