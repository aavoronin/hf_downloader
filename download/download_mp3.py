import os
import re
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def download_alice_mp3_files(index_url: str, output_folder: str,
                             max_retries: int = 3, retry_delay: float = 2.0) -> list[str]:
    """
    Download MP3 audio files from Project Gutenberg Alice audiobook index page.
    Includes retry logic for each individual MP3 file download.

    Args:
        index_url: URL of the HTML index page containing MP3 links
        output_folder: Local folder path to save downloaded MP3 files
        max_retries: Number of retry attempts per file (default: 3)
        retry_delay: Base delay in seconds between retries (default: 2.0)

    Returns:
        List of paths to successfully downloaded files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download and parse the index page
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(index_url, headers=headers, timeout=30)
    response.raise_for_status()
    response.encoding = 'utf-8'

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract MP3 links (relative paths like "mp3/19573-01.mp3")
    mp3_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.mp3'):
            mp3_links.append(href)

    # Build base URL for resolving relative paths
    base_url = re.sub(r'/[^/]+-index\.html?$', '/', index_url)

    downloaded_files = []

    for relative_mp3_url in mp3_links:
        # Construct full URL
        full_url = urljoin(base_url, relative_mp3_url)

        # Extract filename from URL
        filename = Path(urlparse(full_url).path).name
        local_path = output_path / filename

        # Skip if already downloaded
        if local_path.exists():
            print(f"✓ Skipped (exists): {filename}")
            downloaded_files.append(str(local_path))
            continue

        # Retry logic for individual file download
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                print(f"↓ Downloading: {filename} (attempt {attempt}/{max_retries})...")
                with requests.get(full_url, headers=headers, timeout=60, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                print(f"✓ Saved: {local_path}")
                downloaded_files.append(str(local_path))
                break  # Success - exit retry loop

            except requests.RequestException as e:
                last_error = e
                # Clean up partial file on failure
                if local_path.exists():
                    local_path.unlink(missing_ok=True)

                if attempt < max_retries:
                    # Exponential backoff: delay doubles each attempt
                    wait_time = retry_delay * (2 ** (attempt - 1))
                    print(f"⚠ Failed (attempt {attempt}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Failed to download {filename} after {max_retries} attempts: {e}")

    print(f"\n✅ Done: {len(downloaded_files)}/{len(mp3_links)} files downloaded to {output_folder}")
    return downloaded_files

def download_alice():
    # Calling code (place this where you want to execute the download)
    downloaded = download_alice_mp3_files(
        index_url="https://www.gutenberg.org/files/19573/19573-index.html",
        output_folder=r"D:\Data\alice"
    )

    # Optional: print summary
    if downloaded:
        print(f"\nDownloaded files:")
        for f in downloaded:
            print(f"  • {Path(f).name}")