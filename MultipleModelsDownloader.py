import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from HFModelLister import HFModelLister
from hf_downloader import HFModelDownloader


class MultipleModelsDownloader:
    """Downloader that uses HFModelLister to fetch, filter, and save model info."""

    def __init__(self, start_urls: List[str], root_folder: str, force_download_all: bool = False):
        """
        Initialize the downloader.

        Args:
            start_urls: List of HuggingFace model listing URLs to scrape
            root_folder: Root directory for storing model info folders
            force_download_all: If True, force redownload all models regardless of download_date
        """
        self.start_urls = start_urls
        self.root_folder = Path(root_folder)
        self.root_folder.mkdir(parents=True, exist_ok=True)
        self.force_download_all = force_download_all
        self.lister = None

    @staticmethod
    def _passes_filter(model_info: Dict, min_downloads: int = 100, min_likes: int = 10) -> bool:
        """
        Filter models based on downloads and likes thresholds.

        Args:
            model_info: Dict containing model metadata from HFModelLister
            min_downloads: Minimum download count threshold
            min_likes: Minimum like count threshold

        Returns:
            True if model passes filter, False otherwise
        """
        downloads = model_info.get("Downloads", 0)
        likes = model_info.get("Likes", 0)
        return downloads > min_downloads or likes > min_likes

    def _save_model_info(self, model_info: Dict):
        """Save model info to a JSON file in a subfolder. Does not override if file exists."""
        model_id = model_info.get("Model ID", "")
        if not model_id:
            return

        # Create safe folder name from model ID (handle org/model format)
        safe_name = model_id.replace("/", "_")
        model_folder = self.root_folder / safe_name
        model_folder.mkdir(parents=True, exist_ok=True)

        # Save JSON config file - only if it doesn't already exist
        json_path = model_folder / "model_info.json"

        if json_path.exists():
            print(f"⊘ Skipped (exists): {model_id}")
            return

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved: {model_id}")

    def _update_model_info_with_download_date(self, model_info: Dict):
        """Update model_info.json with download_date after successful download."""
        model_id = model_info.get("Model ID", "")
        if not model_id:
            return

        safe_name = model_id.replace("/", "_")
        model_folder = self.root_folder / safe_name
        json_path = model_folder / "model_info.json"

        if json_path.exists():
            # Add or update download_date
            model_info["download_date"] = datetime.now().isoformat()
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            print(f"✓ Updated {model_id} with download_date")

    def _needs_download(self, model_info: Dict) -> bool:
        """Check if model needs to be downloaded based on download_date."""
        if self.force_download_all:
            return True
        # If no download_date, needs download
        return "download_date" not in model_info

    def process_urls(self):
        """Process all start URLs: fetch, filter, and save qualifying models."""
        for start_url in self.start_urls:
            clean_url = start_url.strip()
            print(f"\nProcessing: {clean_url}")

            # Use the existing HFModelLister class
            self.lister = HFModelLister(clean_url)
            self.lister.fetch_all_pages()

            # Filter and save models that pass the threshold
            for model_info in self.lister.results:
                if self._passes_filter(model_info):
                    self._save_model_info(model_info)

    def show_results(self):
        if self.lister is not None:
            self.lister.show_results()

    def download_models(self):
        """Download all local models that haven't been downloaded yet."""
        print(f"\n{'=' * 60}")
        print(f"Downloading models from: {self.root_folder}")
        print('=' * 60)

        # Initialize the HFModelDownloader
        hf_downloader = HFModelDownloader(verbose=True)

        # Get all local models
        local_models = self.list_local_models()

        if not local_models:
            print("No local models found to download.")
            return

        # 2nd loop: load each model using HFModelDownloader
        for model_info in local_models:
            model_id = model_info.get("Model ID", "")
            if not model_id:
                continue

            # Check if download is needed
            if not self._needs_download(model_info):
                print(f"✓ {model_id} already downloaded - skipping")
                continue

            # Build target directory path
            safe_name = model_id.replace("/", "_")
            target_dir = self.root_folder / safe_name

            print(f"\nDownloading: {model_id}")
            print(f"  Target: {target_dir}")

            # Use HFModelDownloader to download the model
            # Pass force_redownload=True if force_download_all is enabled
            success = hf_downloader.download(
                model_id=model_id,
                target_dir=str(target_dir),
                force_redownload=self.force_download_all
            )

            if success:
                # Update model_info.json with download_date
                self._update_model_info_with_download_date(model_info)
            else:
                print(f"✗ Failed to download {model_id}")

    def list_local_models(self) -> List[Dict]:
        """
        List all locally saved models by scanning root folder.

        Returns:
            List of model info dicts loaded from model_info.json files
        """
        local_models = []

        for item in self.root_folder.iterdir():
            if item.is_dir():
                json_path = item / "model_info.json"
                if json_path.exists():
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            model_info = json.load(f)
                            local_models.append(model_info)
                    except json.JSONDecodeError as e:
                        print(f"✗ Error loading {json_path}: {e}")

        return local_models

    def print_local_models(self):
        """Print parameters of all locally saved models."""
        print(f"\n{'=' * 60}")
        print(f"Local Models in: {self.root_folder}")
        print('=' * 60)

        local_models = self.list_local_models()

        if not local_models:
            print("No local models found.")
            return

        for model in local_models:
            print(f"\nModel ID: {model.get('Model ID', 'Unknown')}")
            print(f"  Purpose:      {model.get('Purpose', '')}")
            print(f"  Updated:      {model.get('Updated', '')}")
            print(f"  Downloads:    {model.get('Downloads', 0):,}")
            print(f"  Likes:        {model.get('Likes', 0):,}")
            print(f"  Downloaded:   {model.get('download_date', 'Not yet')}")

        print(f"\n📊 Total: {len(local_models)} models")