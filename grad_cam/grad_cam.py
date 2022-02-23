import cv2
import numpy as np
import torch.nn
from . import utils
import albumentations as A


class GradCAM(object):
    def __init__(self, model, target_layer=None, input_size=None):
        """
        Initialize a Grad-CAM object and initialize its hooks on the model.
        Args:
            model:  The model to explain using Grad-CAM.
            target_layer: A reference to the layer that will be hooked.
                          If this value is None, the last convolutional layer in the model will be chosen.
            input_size: A tuple representing the input size spatial dimensions. If it is None,
                        The returned heatmaps won't be scaled to the input size.
        """
        self.model = model
        self.target_layer = target_layer or self._get_last_conv_layer()

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self._inputs = None
        self._gradients = None
        self._activations = None
        self._handles = []
        self._initialize_hooks()

    def __del__(self):
        for handle in self._handles:
            handle.remove()

    def get_heatmap_projection(self, img_pathes):
        """
        Creates and returns the GradCAM heatmap, projected on the original images that
        were the last input batch to the model.
        Note that this function should be called after each batch, as all of the maps
        which are saved and used by this function are overwritten during each batch.

        Returns:
            A tuple of the form (hm_projections, overlap_scores), where -
                hm_projections - An np.ndarray of size (B, H, W, 3), representing the last batch of inputs, with
                                 the GradCAM heatmap projected on them.
                overlap_scores - An np.ndarray of size (B,) containing the overlapping score for each
                                (image, heatmap) pair.
        """
        # a_k = (1/HW) * Sum(dY/dA_k)
        layer_coefficients = self._gradients.mean(axis=[2, 3])

        # Heatmap = ReLU(Sum(a_k * A_k))
        weighted_layers = layer_coefficients[:, :, None, None] * self._activations
        heatmap = torch.relu(torch.sum(weighted_layers, axis=1))

        # Normalize and resize heatmap, then merge with input image
        heatmap = heatmap.cpu().numpy()
        heatmap = utils.normalize(heatmap)
        heatmap = utils.resize(heatmap, self.input_size)
        merged_images_and_heatmaps = self._merge_images_and_heatmaps(heatmap, img_pathes)

        overlap_scores = utils.get_overlap_scores(self._inputs.permute(0, 2, 3, 1).numpy(), heatmap)

        # Clear the current tensors and prepare the GradCam object for the next batch processing
        self._clear_state()

        return merged_images_and_heatmaps, overlap_scores

    def _clear_state(self):
        # Remove the current state of the GradCAM hooks.
        # This is called by "get_heatmap_projection", which should be called after
        # every batch execution of the model (forward-backward pass).
        self.input_size = None
        self._inputs = None
        self._gradients = None
        self._activations = None

    def _merge_images_and_heatmaps(self, heatmaps, image_paths):
        # Create a weighted combination of the enlarged heatmap and the input image that
        # generated the heatmap
        outputs = []

        rsz = A.Resize(512, 512)

        # image_inputs = np.uint8(self._inputs.permute(0, 2, 3, 1).numpy() * 255)
        image_inputs = [cv2.imread(img_path, cv2.IMREAD_COLOR) for img_path in image_paths]
        image_inputs = np.array([rsz(image=img)['image'] for img in image_inputs], dtype='uint8')
        for i, heatmap in enumerate(heatmaps):
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            merged_image = cv2.addWeighted(heatmap, 0.5, image_inputs[i], 0.5, 0)
            outputs.append(merged_image)
        return np.array(outputs)

    def _get_last_conv_layer(self):
        # Iterate over the layers, starting from the last layer and going backwards,
        # until the first 2-D convolutional layer is found
        modules = list(self.model.named_modules())
        for module_name, module in reversed(modules):
            if isinstance(module, torch.nn.Conv2d):
                return module

        raise RuntimeError('No 2-D Convolution layer found in model. Please supply it yourself'
                           'using the "target_layer" parameter to GradCAM constructor.')

    def _initialize_hooks(self):
        first_layer = list(self.model.named_modules())[0][1]
        self._handles.append(first_layer.register_forward_hook(self._save_model_inputs))
        self._handles.append(self.target_layer.register_forward_hook(self._save_layer_activations))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_layer_gradients))

    def _save_model_inputs(self, module, inputs, outputs):
        # Inputs should be a tensor of the shape (B, 3, H, W) for -
        #   B = batch size
        #   H,W = Spatial dimensions of the model input
        assert self._inputs is None
        if self.input_size is None:
            self.input_size = (inputs[0].shape[2], inputs[0].shape[3])
        self._inputs = inputs[0].cpu().detach()

    def _save_layer_activations(self, module, inputs, outputs):
        # Outputs should be a tensor of the shape (B, C, H, W) for -
        #   B = batch size
        #   C = Channels (number of kernels in the target convolution layer)
        #   H,W = Spatial dimensions of the last layer
        assert self._activations is None
        self._activations = outputs.cpu().detach()

    def _save_layer_gradients(self, module, inputs_grad, outputs_grad):
        # Outputs grad is a tuple of size 1, where the value (dy/dA_ij) should be a tensor of shape (B, C, H, W) for -
        #   B = batch size
        #   C = class count
        #   H,W = Spatial dimensions of the last layer
        assert self._gradients is None
        self._gradients = outputs_grad[0].cpu().detach()
