import os
import sys

# Get the absolute path of the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from ImageHarvester.Scrapper.website_scrapper import (
    HTMLFetcher,
    UrlParser,
    WebsiteScraper,
)


# Test the HTMLFetcher class
def test_fetch_page_content():
    fetcher = HTMLFetcher(
        "https://www.google.com/search?q=cats&sxsrf=APwXEdd8Kx_2LzwwkFQOmP4xDnkq2P8lGQ:1681963407942&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjuj9L7ybf-AhUykFwKHY1jAEQQ_AUoAXoECAEQAw&biw=1920&bih=862&dpr=1#imgrc=eAP244UcF5wdYM"
    )
    content = fetcher.fetch_page_content()
    assert content is not None

    fetcher = HTMLFetcher("https://httpbin.org/status/404")
    content = fetcher.fetch_page_content()
    assert content is None


# Test the UrlParser class
def test_extract_image_urls():
    parser = UrlParser()
    html_content = """
    <html>
        <head></head>
        <body>
            <img src="https://example.com/image1.jpg" />
            <img data-src="https://example.com/image2.jpg" />
            <picture>
                <source srcset="https://example.com/image3.jpg" />
            </picture>
        </body>
    </html>
    """
    image_urls = parser.extract_image_urls(html_content)
    assert len(image_urls) == 3
    assert "https://example.com/image1.jpg" in image_urls
    assert "https://example.com/image2.jpg" in image_urls
    assert "https://example.com/image3.jpg" in image_urls


# Test the ImageHarvester class
def test_image_harvester():
    harvester = WebsiteScraper(
        "https://www.google.com/search?q=cats&sxsrf=APwXEdd8Kx_2LzwwkFQOmP4xDnkq2P8lGQ:1681963407942&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjuj9L7ybf-AhUykFwKHY1jAEQQ_AUoAXoECAEQAw&biw=1920&bih=862&dpr=1#imgrc=eAP244UcF5wdYM"
    )
    image_urls = harvester.scrape()
    # This is a simple test, as the actual URLs will depend on the content of the page.
    # You can add more complex tests if needed.
    assert isinstance(image_urls, list)
