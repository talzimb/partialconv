from albumentations.core.transforms_interface import ImageOnlyTransform


class InverseContrast(ImageOnlyTransform):
    """Inverse image contrast
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        always_apply=False,
        p=0.5,
    ):
        super(InverseContrast, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return 255.0 - img

    def get_transform_init_args_names(self):
        return ()