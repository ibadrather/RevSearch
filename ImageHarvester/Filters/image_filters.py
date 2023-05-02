import imghdr
import io
from PIL import Image
from typing import List
import requests
from tqdm import tqdm


class ImageFilter:
    def __init__(
        self,
        image_urls: List[str],
        file_formats: List[str] = ["jpeg", "png", "gif"],
        min_resolution: int = 512,
        max_resolution: int = 1920,
    ):
        self.image_urls = image_urls
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.file_formats = file_formats

    def filter_by_resolution(self, image: Image.Image) -> bool:
        """Returns True if the image resolution is within the specified range."""
        width, height = image.size

        if width < self.min_resolution or height < self.min_resolution:
            return False

        if width > self.max_resolution or height > self.max_resolution:
            return False

        return True

    def filter_by_file_format(self, image_data: bytes) -> bool:
        """Returns True if the image file format is in the list of allowed formats."""
        file_format = imghdr.what(None, h=image_data)
        return file_format in self.file_formats

    def fetch_image_data(self, image_url: str) -> bytes:
        response = requests.get(image_url)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(
                f"Failed to fetch the image. Status code: {response.status_code}"
            )

    def filter_images(self) -> List[str]:
        """Filters the image URLs based on resolution and file format."""
        filtered_image_urls = []

        for image_url in tqdm(self.image_urls, desc="Filtering images"):
            try:
                image_data = self.fetch_image_data(image_url)

                if not self.filter_by_file_format(image_data):
                    continue

                image = Image.open(io.BytesIO(image_data))

                if not self.filter_by_resolution(image):
                    continue

                filtered_image_urls.append(image_url)
            except Exception as e:
                print(f"Error fetching image: {e}")

        return filtered_image_urls
