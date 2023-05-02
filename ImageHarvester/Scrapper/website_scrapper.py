import requests
from bs4 import BeautifulSoup
from typing import List, Optional


class HTMLFetcher:
    """Fetches the HTML content of a given URL."""

    def __init__(self, url: str):
        self.url = url

    def fetch_page_content(self) -> Optional[str]:
        """Fetches the page content of the URL."""
        try:
            response = requests.get(self.url)
            if response.status_code == 200:
                return response.text
            else:
                print(
                    f"Failed to fetch the content. Status code: {response.status_code}"
                )
                return None
        except Exception as e:
            print(f"Error fetching page content: {e}")
            return None


class UrlParser:
    """Extracts image URLs from HTML content."""

    def __init__(self):
        self.html_parser = "html.parser"
        self.image_tags = ["img", "source"]
        self.image_attributes = ["src", "data-src", "srcset"]

    def extract_image_urls(self, page_content: str) -> List[str]:
        """Extracts image URLs from the given HTML content."""
        soup = BeautifulSoup(page_content, self.html_parser)
        image_urls = []

        for tag in self.image_tags:
            elements = soup.find_all(tag)
            for element in elements:
                for attribute in self.image_attributes:
                    if element.has_attr(attribute):
                        url_values = element[attribute].split(",")
                        for url_value in url_values:
                            url = url_value.strip().split(" ")[0]
                            if url.startswith("http"):
                                image_urls.append(url)

        return image_urls


class WebsiteScraper:
    """Scrapes image URLs from a given website."""

    def __init__(self, url: str):
        self.url = url
        self.fetcher = HTMLFetcher(url)
        self.parser = UrlParser()

    def scrape(self) -> List[str]:
        """Scrapes image URLs from the website."""
        page_content = self.fetcher.fetch_page_content()
        if page_content:
            return self.parser.extract_image_urls(page_content)
        return []


if __name__ == "__main__":
    import os

    os.system("clear")
    url = "https://www.bmwusa.com/all-bmws/sedan.html"
    scraper = WebsiteScraper(url)
    image_urls = scraper.scrape()

    # Total number of images scraped
    print("Total number of images", len(image_urls))

    print(image_urls)
