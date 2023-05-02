"""
    This test file covers three test cases:

    test_filter_by_resolution: tests the filter_by_resolution method with sample images of different resolutions.

    test_filter_by_file_format: tests the filter_by_file_format method with sample images of different file formats.

    test_filter_images: tests the filter_images method using a MagicMock to mock the fetch_image_data method, 
    thus removing the external dependency on the requests library.
"""
import pytest
import io
import os
import sys
from unittest.mock import MagicMock
from PIL import Image
import numpy as np

# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from ImageHarvester.Filters.image_filters import ImageFilter


@pytest.fixture
def sample_images():
    images = {
        "low_res_jpeg": Image.fromarray(
            np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8), "RGB"
        ),
        "high_res_jpeg": Image.fromarray(
            np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8), "RGB"
        ),
        "medium_res_png": Image.fromarray(
            np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8), "RGB"
        ),
    }

    return images


def test_filter_by_resolution(sample_images):
    image_filter = ImageFilter([])
    assert not image_filter.filter_by_resolution(sample_images["low_res_jpeg"])
    assert not image_filter.filter_by_resolution(sample_images["high_res_jpeg"])
    assert image_filter.filter_by_resolution(sample_images["medium_res_png"])


def test_filter_by_file_format(sample_images):
    image_filter = ImageFilter([])

    low_res_jpeg_bytes = io.BytesIO()
    sample_images["low_res_jpeg"].save(low_res_jpeg_bytes, format="JPEG")
    low_res_jpeg_bytes = low_res_jpeg_bytes.getvalue()

    high_res_jpeg_bytes = io.BytesIO()
    sample_images["high_res_jpeg"].save(high_res_jpeg_bytes, format="JPEG")
    high_res_jpeg_bytes = high_res_jpeg_bytes.getvalue()

    medium_res_png_bytes = io.BytesIO()
    sample_images["medium_res_png"].save(medium_res_png_bytes, format="PNG")
    medium_res_png_bytes = medium_res_png_bytes.getvalue()

    assert image_filter.filter_by_file_format(low_res_jpeg_bytes)
    assert image_filter.filter_by_file_format(high_res_jpeg_bytes)
    assert image_filter.filter_by_file_format(medium_res_png_bytes)


def test_filter_images(sample_images):
    image_filter = ImageFilter([])

    medium_res_png_bytes = io.BytesIO()
    sample_images["medium_res_png"].save(medium_res_png_bytes, format="PNG")
    medium_res_png_bytes = medium_res_png_bytes.getvalue()

    image_filter.fetch_image_data = MagicMock(return_value=medium_res_png_bytes)
    image_filter.image_urls = ["https://example.com/medium_res.png"]

    filtered_image_urls = image_filter.filter_images()

    assert len(filtered_image_urls) == 1
    assert filtered_image_urls[0] == "https://example.com/medium_res.png"
