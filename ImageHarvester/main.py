from ImageHarvester.Scrapper.website_scrapper import WebsiteScraper
from ImageHarvester.Filters.image_filters import ImageFilter
from ImageHarvester.Downloader.image_downloader import ImageDownloader

if __name__ == "__main__":
    import os

    os.system("clear")

    # Scrape image URLs from the website
    url = "https://www.google.com/search?sxsrf=APwXEdd7AmYvbxJXJI9QkqfFMUzCh9DH-g:1681967189464&source=lnms&tbm=isch&stick=H4sIAAAAAAAAAONQF5LWr0jW18_VNzAyzC0wskpOLFLIzE1MTy12SM1TwiepJZebWpJopV9RkJiXmgNUmJ-XnFpQYlWcmZJanlhZHCVQnJqSmKeA0LSIUQBkXlFqWmZeam5qXom-wS9GQUefYH-FYFfHIGcPVxcFN_-gBSwcC1hYF7AwLWBhX8DCuICFeQELwwIWtkWsAsFoJt5ik2QoeXZtltzzd1Y_nYp6BAOdL52v4Glfwpr7HQA8vkIp3AAAAA&q=Sedan%20car%20images&sa=X&ved=2ahUKEwiUgeiG2Lf-AhWQyqQKHaX5Ct0Q_AUoAXoECAEQAw&biw=1920&bih=862&dpr=1#imgrc=XupwFp-Tj0I_LM"
    scraper = WebsiteScraper(url)
    image_urls = scraper.scrape()

    # Filter the images based on resolution and file format
    file_formats = ["jpg", "jpeg", "png"]
    min_resolution = 224
    max_resolution = 1920

    # before filtering
    print(f"Total number of images before filtering: {len(image_urls)}")

    image_filter = ImageFilter(image_urls, file_formats, min_resolution, max_resolution)
    filtered_image_urls = image_filter.filter_images()

    # Print the filtered image URLs
    print(f"Total number of images after filtering: {len(filtered_image_urls)}")

    output_directory = "downloaded_images/sedan"

    os.makedirs(output_directory, exist_ok=True)

    downloader = ImageDownloader(
        filtered_image_urls,
        output_directory,
        multi_threading=True,
        max_retries=3,
        timeout=10,
    )
    downloader.download_images()
