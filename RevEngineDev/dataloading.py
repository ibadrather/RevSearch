from torch.utils.data import Dataset
import pandas as pd
import torch
import cv2
import numpy as np
import os
# label encoder
from sklearn.preprocessing import LabelEncoder


class StanfordCarDataset(Dataset):
    """
    This class is used to load the data using the csv file.
    The csv file contains the image path and label.

    We can also perform data augmentation using the albumentations library.

    Args:
        csv_file (str): Path to the csv file
        dataset_dir (str): Path to the dataset directory
        config (dict, optional): Configuration dictionary. Defaults to None.

    Parameters:
        data (np.ndarray): Array containing the image path and label
        dataset_dir (str): Path to the dataset directory
        image_size (tuple): Size of the image. Defaults to (224, 224)
        data_augmentation (bool): Whether to perform data augmentation or not. Defaults to False

    Methods:
        __len__(): Returns the length of the dataset
        __getitem__(idx): Returns the image and label at the given index

    Returns:
        tuple: Image and label


    """

    def __init__(
        self, csv_file: str, dataset_dir: str, transforms=None, config: dict = None
    ) -> None:
        # Read the csv file and store the image and mask paths
        self.data = pd.read_csv(csv_file)

        self.image_paths = self.data["filename"].values
        self.labels = self.data["class"].values

        self.dataset_dir = dataset_dir

        # Label encoder
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

        # count the number of classes
        self.num_classes = len(self.label_encoder.classes_)

        if config is None:
            self.image_size = (224, 224)
            if transforms is None:
                self.data_augmentation = False
            elif transforms is not None:
                self.data_augmentation = True
                self.transforms = transforms
        else:
            self.image_size = config["image_size"]
            if transforms is not None and config["data_augmentation"] is True:
                self.data_augmentation = config["data_augmentation"]
                self.transforms = transforms
            else:
                self.data_augmentation = False

    def __len__(self):
        return len(self.data)
    
    # inverse reverse label encoder
    def inverse_transform(self, labels):
        return self.label_encoder.inverse_transform(labels)
    
    def num_classes(self):
        return self.num_classes

    def __getitem__(self, idx) -> tuple:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image
        image = load_image(
            os.path.join(self.dataset_dir, image_path),
            size=self.image_size,
            normalize_image=False,
        )

        # Data augmentation
        if self.data_augmentation:
            augmented = self.transforms(
                image=image,
            )
            image = augmented["image"]

        # Normalize the image
        image = image / 255.0

        # -1 to + 1 normalization
        image = (image - 0.5) / 0.5

        return torch.Tensor(image).permute(2, 0, 1), torch.tensor(label).long()


def load_image(
    image_path: str,
    size: tuple = (224, 224),
    normalize_image: bool = True,
) -> np.ndarray:
    """
    This function is used to load the image and mask and resize them.
    Args:
        image_path (str): Path to the image
        size (tuple, optional): Size of the image. Defaults to (224, 224).
        normalize_image (bool, optional): Whether to normalize the image or not. Defaults to True.
    Returns:
        np.ndarray: Image
    """

    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to RGB as opencv reads the image in BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the image if normalize_image is True
    if normalize_image:
        # Normalize the image
        image = image / 255.0

    # Resize the image and mask
    image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

    return image


def display_image(image) -> None:
    """
    This function is used to display the image.
    Args:
        image (np.ndarray): Image to be displayed
    """
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    # first permute axis and convert both to numpy arrays
    image = image.permute(1, 2, 0).numpy()

    # if image data type is not np.uint8 and is not in range 0-255, convert it: i.e. if it is not normalized
    if image.dtype != np.uint8 and np.max(image) <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    plt.imshow(image)
    plt.show()


def test():
    import os

    os.system("clear")

    data_path = "/home/ibad/Desktop/RevSearch/Car196_Combined/images"

    train_csv = os.path.join(
        data_path, "train.csv"
    )

    config = dict(
        image_size=(224, 224),
        data_augmentation=False,
    )

    dataset = StanfordCarDataset(train_csv, data_path, config=config, transforms=None)

    image, label = dataset[0]

    print(image.shape, label)

    display_image(image)


if __name__ == "__main__":
    test()
