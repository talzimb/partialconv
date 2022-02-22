import os
import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import morphology


def get_prediction_strings(y_pred, y):
    """
    Generate a prediction string (TP, FP, TN, FN) for a batch of data.

    Args:
        y_pred (Torch.tensor): The output probabilities for a given set of inputs.
        y      (Torch.tensor): The true classifications for a given set of inputs.
    Returns:
        List(str): A list containing the prediction strings, each representing one
                entry in the input tensors.
    """
    predictions = torch.argmax(y_pred, dim=1)
    prediction_strings = []
    for i in range(len(predictions)):
        prediction_strings.append('%s%s' % ('T' if predictions[i] == y[i] else 'F',
                                            'P' if predictions[i] == 1 else 'N'))
    return prediction_strings


def normalize(heatmaps):
    # Normalize the heatmap to contain values between [0, 1]
    # in a manner that makes the heatmap look good.
    flattened_heatmaps = heatmaps.reshape([heatmaps.shape[0], -1])
    min_values = np.min(flattened_heatmaps, 1)[:, None, None]
    max_values = np.max(flattened_heatmaps, 1)[:, None, None]
    return (heatmaps - min_values) / (max_values - min_values + np.finfo(heatmaps.dtype).eps)


def resize(heatmaps, output_size):
    # Enlarge the heatmap generated from the last convolutional layer to the size
    # of the original image using cv2 default interpolation
    return np.array([cv2.resize(h, output_size) for h in heatmaps])


def get_overlap_scores(images, heatmaps):
    """
    Get, for each image and corresponding heatmap, the percentage of the heatmap
    that lays within the relevant areas of classification (i.e. - inside the lungs).

    Args:
        images   (List(np.ndarray)): The list of images.
        heatmaps (List(np.ndarray)): The list of heatmaps of the images.

    Returns:
        np.ndarray: An array containing the overlapping score for each (image, heatmap) pair.
    """
    assert heatmaps.shape == images.shape[:3] and images.shape[3] == 3

    # This should give a high score when
    # Heatmap warm areas are within lungs
    scores = []
    for i in range(len(heatmaps)):
        # Generate the mask of the image (based on intensities)
        mask = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
        mask = morphology.dilation(mask, morphology.disk(7))

        # Get the "ReLU-ed" version of the heat
        thresholded_heatmap = heatmaps[i]
        thresholded_heatmap[thresholded_heatmap < 0.25] = 0

        unified = (thresholded_heatmap * mask)

        # Get the overlap score as the percentage of the heatmap which was inside the mask
        overlap_score = unified.sum() / (thresholded_heatmap.sum() + 1e-5)
        scores.append(overlap_score)

    return np.array(scores)


def save_images(heatmap_images, file_names, output_dir):
    assert len(heatmap_images) == len(file_names), "The image list and file name list must be of the same size"
    os.makedirs(output_dir, exist_ok=True)
    for i, h in enumerate(heatmap_images):
        image_path = osp.join(output_dir, file_names[i])
        Image.fromarray(h).save(image_path)
