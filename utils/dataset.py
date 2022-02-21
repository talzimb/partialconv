import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, file_paths, masks_paths, dataframe, transform=None):
        self.file_paths = file_paths # path to images
        self.masks_paths = masks_paths
        self.labels_frame = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # label from data frame - positive / negative
        # label = self.labels_frame.iloc[idx, 1]
        class_to_idx = {'negative': 0, "positive": 1}
        label = class_to_idx[self.labels_frame.iloc[idx]]

        file_path = self.file_paths[idx]

        image = Image.open(file_path)
        mask = Image.open(self.masks_paths[idx])

        if self.transform:
            # Convert PIL image to numpy array
            image_np = np.uint8(np.array(image.convert('RGB')).astype(np.uint8))
            mask_np = np.array(mask)
            # Apply transformations
            augmented = self.transform(image=image_np, mask=mask_np)
            # Convert numpy array to PIL Image
            # image = Image.fromarray(augmented['image'])
            image = augmented['image']
            mask = augmented['mask']
        return image, label, mask
    #
    # def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
    #     """Find the class folders in a dataset structured as follows::
    #
    #         directory/
    #         ├── class_x
    #         │   ├── xxx.ext
    #         │   ├── xxy.ext
    #         │   └── ...
    #         │       └── xxz.ext
    #         └── class_y
    #             ├── 123.ext
    #             ├── nsdf3.ext
    #             └── ...
    #             └── asd932_.ext
    #
    #     This method can be overridden to only consider
    #     a subset of classes, or to adapt to a different dataset directory structure.
    #
    #     Args:
    #         directory(str): Root directory path, corresponding to ``self.root``
    #
    #     Raises:
    #         FileNotFoundError: If ``dir`` has no class folders.
    #
    #     Returns:
    #         (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
    #     """
    #     return find_classes(directory)
