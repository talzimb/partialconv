import albumentations as A
from utils.custom_augs import InverseContrast


def data_transforms(phase=None, mask=None):
    if phase == 'TRAIN':

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])

        data_T = A.Compose([A.Resize(512, 512),
                 A.HorizontalFlip(p=0.5),
                 A.Rotate([-10, 10], p=0.6),
                 A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                 InverseContrast(p=0.3),
                 A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5),
                 # A.Equalize(mode='cv', by_channels=True, mask=mask, mask_params=(), always_apply=False, p=0.5),
                 A.ToFloat(max_value=255.0),
                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    elif phase == 'TEST' or phase == 'VAL':

        data_T = A.Compose([A.Resize(512, 512),
                            A.ToFloat(max_value=255.0),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return data_T