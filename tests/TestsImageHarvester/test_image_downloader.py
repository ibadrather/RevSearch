import os
import sys
import io
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from ImageHarvester.Downloader.image_downloader import ImageDownloader


@pytest.fixture
def image_data():
    image = Image.fromarray(
        np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8), "RGB"
    )
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def test_download_image(image_data, tmpdir):
    image_url = "http://example.com/image.jpg"

    # Mock requests.get
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = image_data

    output_directory = str(tmpdir.mkdir("downloaded_images"))
    downloader = ImageDownloader([image_url], output_directory)

    with patch("requests.get", return_value=mock_response):
        assert downloader.download_image(image_url)

    assert len(os.listdir(output_directory)) == 1
    assert os.path.isfile(os.path.join(output_directory, "image.jpg"))


def test_download_images(image_data, tmpdir):
    image_urls = [
        "http://example.com/image1.jpg",
        "http://example.com/image2.jpg",
        "http://example.com/image3.jpg",
    ]

    # Mock requests.get
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = image_data

    output_directory = str(tmpdir.mkdir("downloaded_images"))
    downloader = ImageDownloader(image_urls, output_directory)

    with patch("requests.get", return_value=mock_response):
        downloader.download_images()

    assert len(os.listdir(output_directory)) == 3
    for img_url in image_urls:
        file_name = img_url.split("/")[-1]
        assert os.path.isfile(os.path.join(output_directory, file_name))
