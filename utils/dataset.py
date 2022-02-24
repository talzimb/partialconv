import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, file_paths, masks_paths, dataframe, phase, transform=None):
        self.file_paths = file_paths # path to images
        self.masks_paths = masks_paths
        self.labels_frame = dataframe
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # label from data frame - positive / negative
        # label = self.labels_frame.iloc[idx, 1]
        class_to_idx = {'negative': 0, "positive": 1}
        label = class_to_idx[self.labels_frame.iloc[idx]]

        file_path = self.file_paths[idx]
        img_name = file_path.split('/')[-1]
        img_name = img_name[:img_name.rfind('.')]  #remove file type from image name

        image = Image.open(file_path)
        mask = Image.open(self.masks_paths[idx])
        mask = self.resize_mask(mask, image)

        if self.transform:
            # Convert PIL image to numpy array
            image_np = np.uint8(np.array(image.convert('RGB')).astype(np.uint8))
            mask_np = np.array(mask)
            # if self.phase == 'TRAIN':
            #     image_np = self.equalize_img(image_np, mask_np)
            # Apply transformations
            augmented = self.transform(image=image_np, mask=mask_np)
            # Convert numpy array to PIL Image
            # image = Image.fromarray(augmented['image'])
            image = augmented['image']
            mask = augmented['mask']
        return image, label, mask, file_path

    def resize_mask(self, mask, image):
        im = np.array(image)
        if len(im.shape) == 3:
            c, w, h = im.shape
        else:
            w, h = im.shape
        # resize mask to image dimension
        Re_tra = A.Compose([
            A.Resize(width=w, height=h),
        ])
        re_mask = Re_tra(image=np.array(mask))['image']
        return Image.fromarray(re_mask)

    def equalize_img(self, image, mask):
        equ_tr = A.Compose([A.Equalize(mode='cv', mask=mask, by_channels=False)])
        eq_image = equ_tr(image=image)['image']

        return eq_image