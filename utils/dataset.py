import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        image = Image.open(file_path)

        if self.transform:
            # Convert PIL image to numpy array
            image_np = np.array(image).astype(np.uint8)
            # Apply transformations
            augmented = self.transform(image=image_np)
            # Convert numpy array to PIL Image
            image = Image.fromarray(augmented['image'])
        return image, label