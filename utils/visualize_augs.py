import copy
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image


def visualize_augmentations(dataset, transform, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    # dataset.transform = transforms
    dataset.transform = A.Compose([t for t in transform if not isinstance(t, (A.Normalize, A.ToFloat))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _, mask = dataset[idx]
        ax.ravel()[i].imshow(image, cmap='gray')
        ax.ravel()[i+1].imshow(mask, cmap='gray')
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def dataset_loop(dataset):
    for i, data in enumerate(dataset):
        file = data
