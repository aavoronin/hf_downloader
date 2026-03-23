import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from HFModelLister import HFModelLister


# Import the existing, working HFModelLister class
# from your_module import HFModelLister


class MultipleModelsDownloader:
    """Downloader that uses HFModelLister to fetch, filter, and save model info."""

    def __init__(self, start_urls: List[str], root_folder: str):
        """
        Initialize the downloader.

        Args:
            start_urls: List of HuggingFace model listing URLs to scrape
            root_folder: Root directory for storing model info folders
        """
        self.start_urls = start_urls
        self.root_folder = Path(root_folder)
        self.root_folder.mkdir(parents=True, exist_ok=True)

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
        """Save model info to a JSON file in a subfolder."""
        model_id = model_info.get("Model ID", "")
        if not model_id:
            return

        # Create safe folder name from model ID (handle org/model format)
        safe_name = model_id.replace("/", "_")
        model_folder = self.root_folder / safe_name
        model_folder.mkdir(parents=True, exist_ok=True)

        # Save JSON config file
        json_path = model_folder / "model_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved: {model_id}")

    def process_urls(self):
        """Process all start URLs: fetch, filter, and save qualifying models."""
        for start_url in self.start_urls:
            clean_url = start_url.strip()
            print(f"\nProcessing: {clean_url}")

            # Use the existing HFModelLister class
            lister = HFModelLister(clean_url)
            lister.fetch_all_pages()

            # Filter and save models that pass the threshold
            for model_info in lister.results:
                if self._passes_filter(model_info):
                    self._save_model_info(model_info)

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
            print(f"  Purpose:   {model.get('Purpose', '')}")
            print(f"  Updated:   {model.get('Updated', '')}")
            print(f"  Downloads: {model.get('Downloads', 0):,}")
            print(f"  Likes:     {model.get('Likes', 0):,}")

        print(f"\n📊 Total: {len(local_models)} models")


