import numpy as np
from PIL import Image


def preprocess_fn(image: Image.Image, input_size: tuple) -> np.ndarray:
    """
    Preprocess an image for ONNX model inference.

    Args:
        image (PIL.Image.Image): The input image.
        input_size (tuple): The desired size of the input image (width, height).

    Returns:
        numpy.ndarray: The preprocessed image as a NumPy array.
    """
    img = image.convert("RGB")
    img = img.resize(input_size, Image.BILINEAR)
    img_data = np.array(img).transpose(2, 0, 1).astype(np.float32)
    img_data /= 255.0
    img_data = (img_data - 0.5) / 0.5

    return img_data[np.newaxis, :, :, :]
