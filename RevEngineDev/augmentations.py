from typing import Optional, Tuple, Dict, Callable

import albumentations as A


class AlbumentationsDataAugmentation:
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        options: Optional[Dict[str, bool]] = None,
    ) -> None:
        """
        Data augmentation class based on Albumentations library.

        Args:
            image_size (Tuple[int, int], optional): Size of the image. Defaults to (224, 224).
            options (Optional[Dict[str, bool]], optional): Dictionary containing augmentation options. Defaults to None.
        """
        self.image_size = image_size
        self.options = options or {}
        self.transform = self.create_transform()

    def create_transform(self) -> A.Compose:
        """
        Creates a transform pipeline based on the selected options.

        Returns:
            A.Compose: Augmentations pipeline.
        """
        transforms = []

        # Resize the image
        transforms.append(A.Resize(*self.image_size))

        # Light non-destructive augmentations
        self._add_light_augmentations(transforms)

        # Non-rigid transformations and RandomSizedCrop
        self._add_medium_augmentations(transforms)

        # Non-spatial transformations
        self._add_non_spatial_transformations(transforms)

        return A.Compose(transforms)

    def _add_light_augmentations(self, transforms: list) -> None:
        if self.options.get("horizontal_flip", True):
            transforms.append(A.HorizontalFlip(p=0.5))

        if self.options.get("vertical_flip", True):
            transforms.append(A.VerticalFlip(p=0.5))

        if self.options.get("random_rotate_90", True):
            transforms.append(A.RandomRotate90(p=0.5))

        if self.options.get("transpose", True):
            transforms.append(A.Transpose(p=0.5))

    def _add_medium_augmentations(self, transforms: list) -> None:
        if self.options.get("medium_augmentations", True):
            transforms.append(
                A.CenterCrop(p=0.5, height=self.image_size[1], width=self.image_size[0])
            )
            transforms.append(
                A.OneOf(
                    [
                        A.PadIfNeeded(
                            min_height=self.image_size[1],
                            min_width=self.image_size[0],
                            p=0.5,
                        ),
                        A.RandomSizedCrop(
                            min_max_height=(
                                self.image_size[1] // 2,
                                self.image_size[1],
                            ),
                            height=self.image_size[0],
                            width=self.image_size[1],
                            p=0.5,
                        ),
                    ],
                    p=1,
                )
            )
            transforms.append(
                A.OneOf(
                    [
                        A.ElasticTransform(
                            p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                        ),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
                    ],
                    p=0.5,
                )
            )

    def _add_non_spatial_transformations(self, transforms: list) -> None:
        if self.options.get("clahe", True):
            transforms.append(A.CLAHE(p=0.5))

        if self.options.get("random_brightness_contrast", True):
            transforms.append(A.RandomBrightnessContrast(p=0.5))

        if self.options.get("random_gamma", True):
            transforms.append(A.RandomGamma(p=0.5))

    def __call__(self, image: Dict[str, any]) -> Dict[str, any]:
        """
        Apply the data augmentation pipeline to the given image.

        Args:
            image (Dict[str, any]): Dictionary containing image data. Expected to have a key 'image' with image array as its value.

        Returns:
            Dict[str, any]: Dictionary containing the augmented image data.
        """
        return self.transform(image=image)
