import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from download.HFModelLister import HFModelLister
from download.hf_downloader import HFModelDownloader


class MultipleModelsDownloader:
    """Downloader that uses HFModelLister to fetch, filter, and save model info."""

    def __init__(self, start_urls: List[str], root_folder: str,
                 force_download_all: bool = False, exclude: List[str] = []):
        """
        Initialize the downloader.

        Args:
            start_urls: List of HuggingFace model listing URLs to scrape
            root_folder: Root directory for storing model info folders
            force_download_all: If True, force redownload all models regardless of download_date
        """
        self.exclude = exclude
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

    def _get_model_json_path(self, model_id: str) -> Path:
        """Helper to get the path to the model_info.json file."""
        safe_name = model_id.replace("/", "_")
        model_folder = self.root_folder / safe_name
        return model_folder / "model_info.json"

    def _save_model_info(self, model_info: Dict):
        """Save model info to a JSON file in a subfolder. Does not override if file exists."""
        model_id = model_info.get("Model ID", "")
        if not model_id:
            return

        json_path = self._get_model_json_path(model_id)
        model_folder = json_path.parent
        model_folder.mkdir(parents=True, exist_ok=True)

        if json_path.exists():
            print(f"⊘ Skipped (exists): {model_id}")
            return

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved: {model_id}")

    def _update_model_config(self, model_id: str, updates: Dict):
        """
        Generic helper to update the model_info.json with specific fields.

        Args:
            model_id: The HuggingFace model ID
            updates: Dictionary of fields to update/add
        """
        json_path = self._get_model_json_path(model_id)

        if not json_path.exists():
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                model_info = json.load(f)

            model_info.update(updates)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, IOError) as e:
            print(f"✗ Error updating config for {model_id}: {e}")

    def _update_model_info_with_download_date(self, model_id: str):
        """Update model_info.json with download_date and reset failed_attempts on success."""
        updates = {
            "download_date": datetime.now().isoformat(),
            "failed_attempts": 0
        }
        self._update_model_config(model_id, updates)
        print(f"✓ Updated {model_id} with download_date and reset failed_attempts")

    def _increment_failed_attempts(self, model_id: str):
        """Increment the failed_attempts counter in the config file."""
        json_path = self._get_model_json_path(model_id)

        if not json_path.exists():
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                model_info = json.load(f)

            current_attempts = model_info.get("failed_attempts", 0)
            new_attempts = current_attempts + 1

            model_info["failed_attempts"] = new_attempts

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)

            print(f"⚠ Failed attempts for {model_id} incremented to {new_attempts}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"✗ Error updating failed_attempts for {model_id}: {e}")

    def _needs_download(self, model_info: Dict) -> bool:
        """Check if model needs to be downloaded based on download_date."""
        if self.force_download_all:
            return True
        # If no download_date, needs download
        return "download_date" not in model_info

    def _is_blocked_by_failures(self, model_info: Dict) -> bool:
        """Check if model should be skipped due to too many failed attempts."""
        if self.force_download_all:
            return False

        failed_attempts = model_info.get("failed_attempts", 0)
        return failed_attempts > 5

    def _calculate_folder_stats(self, model_id: str) -> Dict:
        """
        Calculate folder statistics: total size in bytes, human-readable size, and file count.
        Excludes .nach files and model_info.json from calculation.

        Args:
            model_id: The HuggingFace model ID

        Returns:
            Dict with 'size', 'size_str', and 'numfiles'
        """
        safe_name = model_id.replace("/", "_")
        model_folder = self.root_folder / safe_name

        total_size = 0
        num_files = 0

        for file_path in model_folder.rglob("*"):
            if file_path.is_file():
                # Exclude .nach files and model_info.json
                if file_path.name.endswith(".nach") or file_path.name == "model_info.json":
                    continue
                total_size += file_path.stat().st_size
                num_files += 1

        # Create human-readable size string
        size_str = self._format_size(total_size)

        return {
            "size": total_size,
            "size_str": size_str,
            "numfiles": num_files
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """
        Convert bytes to human-readable format (B, KB, MB, GB, TB).

        Args:
            size_bytes: Size in bytes

        Returns:
            Human-readable size string
        """
        if size_bytes == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        size = float(size_bytes)

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.2f} {units[unit_index]}"

    def _update_model_info_with_folder_stats(self, model_id: str):
        """
        Update model_info.json with folder statistics if they don't exist.
        Checks for 'size', 'size_str', or 'numfiles' fields.
        """
        json_path = self._get_model_json_path(model_id)

        if not json_path.exists():
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                model_info = json.load(f)

            # Check if any of the stats fields are missing
            if not all(key in model_info for key in ["size", "size_str", "numfiles"]):
                stats = self._calculate_folder_stats(model_id)
                model_info.update(stats)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(model_info, f, indent=2, ensure_ascii=False)

                print(f"✓ Updated {model_id} with folder stats: {stats['size_str']}, {stats['numfiles']} files")
        except (json.JSONDecodeError, IOError) as e:
            print(f"✗ Error updating folder stats for {model_id}: {e}")

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

            # Skip models whose ID is in the exclude list
            if model_id in self.exclude:
                print(f"⊘ Excluded: {model_id}")
                continue

            # Check if blocked by failed attempts (> 5)
            if self._is_blocked_by_failures(model_info):
                print(f"⊘ Skipped (failed_attempts > 5): {model_id}")
                continue

            # Check if download is needed (based on date)
            if not self._needs_download(model_info):
                print(f"✓ {model_id} already downloaded - skipping")
                # Update folder stats for previously downloaded models
                self._update_model_info_with_folder_stats(model_id)
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
                # Update model_info.json with download_date and reset attempts
                self._update_model_info_with_download_date(model_id)
                # Update folder stats after successful download
                self._update_model_info_with_folder_stats(model_id)
            else:
                print(f"✗ Failed to download {model_id}")
                # Increment failed attempts in config
                self._increment_failed_attempts(model_id)

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
            # Print Model ID if it exists and has a value
            if model.get('Model ID'):
                print(f"\nModel ID: {model['Model ID']}")

            # Print Purpose if it exists and is not empty
            if model.get('Purpose'):
                print(f"  Purpose:      {model['Purpose']}")

            # Print Updated if it exists and is not empty
            if model.get('Updated'):
                print(f"  Updated:      {model['Updated']}")

            # Print Downloads if the key exists in the dict
            if 'Downloads' in model:
                print(f"  Downloads:    {model['Downloads']:,}")

            # Print Likes if the key exists in the dict
            if 'Likes' in model:
                print(f"  Likes:        {model['Likes']:,}")

            # Print Downloaded if it exists and is not empty
            if model.get('download_date'):
                print(f"  Downloaded:   {model['download_date']}")

            # Print Failures if the key exists in the dict
            if 'failed_attempts' in model:
                print(f"  Failures:     {model['failed_attempts']}")

            # Print Size if the key exists in the dict
            if 'size_str' in model:
                print(f"  Size:         {model['size_str']}")

            # Print NumFiles if the key exists in the dict
            if 'numfiles' in model:
                print(f"  Files:        {model['numfiles']}")

        print(f"\n📊 Total: {len(local_models)} models")

    def print_download_summary(self):
        """Print summary of downloaded models sorted by size descending."""
        local_models = self.list_local_models()

        # Filter models that have calculated sizes
        models_with_size = [m for m in local_models if 'size' in m]

        # Sort by size descending
        sorted_models = sorted(models_with_size, key=lambda x: x.get('size', 0), reverse=True)

        print(f"\n{'=' * 60}")
        print(f"Download Summary (Sorted by Size Desc)")
        print('=' * 60)

        for model in sorted_models:
            model_id = model.get('Model ID', 'Unknown')
            size_str = model.get('size_str', 'Unknown')
            print(f"{model_id}  {size_str}")

        print(f"\n📊 Total Models with Size Info: {len(sorted_models)}")