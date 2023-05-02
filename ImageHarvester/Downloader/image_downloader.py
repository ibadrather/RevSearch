import os
from typing import List
import requests
from concurrent.futures import ThreadPoolExecutor


class ImageDownloader:
    def __init__(
        self,
        image_urls: List[str],
        output_directory: str,
        multi_threading: bool = True,
        max_retries: int = 3,
        timeout: int = 10,
    ):
        self.image_urls = image_urls
        self.output_directory = output_directory
        self.multi_threading = multi_threading
        self.max_retries = max_retries
        self.timeout = timeout

    def download_image(self, image_url: str) -> bool:
        """Downloads an image from the given URL."""
        retries = 0

        while retries < self.max_retries:
            try:
                response = requests.get(image_url, timeout=self.timeout)
                if response.status_code == 200:
                    image_data = response.content
                    file_name = image_url.split("/")[-1]
                    file_path = os.path.join(self.output_directory, file_name)

                    with open(file_path, "wb") as file:
                        file.write(image_data)

                    print(f"Downloaded: {image_url}")
                    return True
                else:
                    print(
                        f"Failed to download: {image_url}. Status code: {response.status_code}"
                    )
            except Exception as e:
                print(f"Error downloading image: {e}")

            retries += 1

        return False

    def download_images(self) -> None:
        """Downloads all the images from the provided image URLs."""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        if self.multi_threading:
            with ThreadPoolExecutor() as executor:
                executor.map(self.download_image, self.image_urls)
        else:
            for image_url in self.image_urls:
                self.download_image(image_url)
