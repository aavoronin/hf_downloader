# hf_downloader.py
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union
from huggingface_hub import snapshot_download, login, list_repo_refs, HfApi
from packaging import version as pkg_version

from download.my_token import load_hf_token


class HFModelDownloader:
    """
    A class for downloading models from Hugging Face Hub.

    Handles authentication, path normalization, download logic,
    and progress reporting internally. Simple two-line usage.
    """

    # === CLASS-LEVEL DEFAULTS ===
    DEFAULT_IGNORE_PATTERNS = ["*.msgpack", "*.ot"]
    DEFAULT_FORCE_REDOWNLOAD = False
    DEFAULT_CREATE_DIR = True
    DEFAULT_ALLOW_GIT_CREDENTIAL = False
    DEFAULT_VERBOSE = True

    def __init__(self, verbose: bool = True):
        """
        Initialize the downloader.

        Token is handled internally - no need to pass it here.

        Args:
            verbose: Enable/disable console output
        """
        self.verbose = verbose
        self._token = None
        self._authenticated = False
        self.try_version = False

    def _log(self, message: str) -> None:
        """Internal logging method respecting verbose flag."""
        if self.verbose:
            print(message)

    def _get_token(self) -> Optional[str]:
        """
        Retrieve token from multiple sources with fallback.

        Priority:
        1. my_token.HF_TOKEN (project config)
        2. HF_TOKEN environment variable
        3. None (let huggingface_hub use cached credentials)
        """
        if self._token is not None:
            return self._token

        self._token = load_hf_token()
        return self._token

    def _authenticate(self) -> bool:
        """Handle Hugging Face authentication."""
        if self._authenticated:
            return True

        token = self._get_token()

        self._log("\n🔐 HuggingFace Authentication")
        self._log("-" * 70)

        if token:
            try:
                login(token=token, add_to_git_credential=self.DEFAULT_ALLOW_GIT_CREDENTIAL)
                self._log("✓ Successfully authenticated with HuggingFace")
                self._authenticated = True
                return True
            except Exception as e:
                self._log(f"⚠ Authentication warning: {e}")
                return False
        else:
            self._log("⚠️  WARNING: No valid token found!")
            self._log("   Ensure my_token.py has HF_TOKEN or set HF_TOKEN env variable")
            return False

    @staticmethod
    def normalize_model_path(path: str) -> str:
        r"""
        Converts WSL/Linux style path to Windows format if running on Windows.
        """
        if os.name == 'nt':
            wsl_pattern = r'^/mnt/([a-zA-Z])(.*)'
            match = re.match(wsl_pattern, path)
            if match:
                drive_letter = match.group(1).upper()
                path_rest = match.group(2).replace('/', '\\')
                return f"{drive_letter}:{path_rest}"
            return path.replace('/', '\\')
        return path

    @staticmethod
    def _generate_label_from_model_id(model_id: str) -> str:
        """Generate a friendly display label from model_id."""
        return model_id.split("/")[-1] if "/" in model_id else model_id

    def _get_directory_size_gb(self, directory: Union[str, Path]) -> float:
        """Calculate total size of directory in GB."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0.0
        total_bytes = sum(
            f.stat().st_size
            for f in dir_path.rglob('*')
            if f.is_file()
        )
        return total_bytes / (1024 ** 3)

    def _check_existing(self, local_dir: str, force_redownload: bool) -> Optional[float]:
        """Check if model already exists and should be skipped."""
        config_path = Path(local_dir) / "model_index.json"
        fallback_checks = [
            Path(local_dir) / "config.json",
            Path(local_dir) / "pytorch_model.bin",
            Path(local_dir) / "model.safetensors",
        ]

        model_exists = config_path.exists() or any(p.exists() for p in fallback_checks)

        if model_exists and not force_redownload:
            size_gb = self._get_directory_size_gb(local_dir)
            return size_gb
        return None

    def _get_best_revision(self, model_id: str, repo_type: str = "model") -> Optional[str]:
        """
        Find the best revision (git tag) to download for a model.

        Looks for semantic version tags (e.g., v23, v11.1, 1.2.3) and returns
        the highest version. Returns None if no valid version tags found.

        Args:
            model_id: HuggingFace repository ID
            repo_type: Type of repo ("model", "dataset", or "space")

        Returns:
            str: Best revision name to use, or None to use default
        """
        try:
            refs = list_repo_refs(repo_id=model_id, repo_type=repo_type)
        except Exception as e:
            self._log(f"  ⚠ Could not fetch revisions for {model_id}: {e}")
            return None

        if not refs or not refs.tags:
            self._log(f"  ℹ No git tags found for {model_id}")
            return None

        # Flexible pattern to match version tags:
        # v23, v11.1, 1.2.3, v18.1-rc1, etc.
        version_pattern = re.compile(r'^v?(\d+(?:\.\d+)*(?:[-+].*)?)$', re.IGNORECASE)
        valid_versions = []

        for tag in refs.tags:
            tag_name = tag.name
            match = version_pattern.match(tag_name)
            if match:
                try:
                    ver = pkg_version.parse(tag_name)
                    valid_versions.append((ver, tag_name))
                except Exception:
                    continue

        if not valid_versions:
            self._log(f"  ℹ No valid version tags found for {model_id}")
            return None

        # Sort by version (highest first) and return the tag name
        valid_versions.sort(key=lambda x: x[0], reverse=True)
        best_tag = valid_versions[0][1]
        self._log(f"  ℹ Found version tags, selecting latest: {best_tag}")
        return best_tag

    def _get_latest_version_files(self, model_id: str, repo_type: str = "model") -> Optional[List[str]]:
        """
        Find files with the highest version number in their filenames.

        This handles cases where versions are in file names (not git tags).
        E.g., model-v23.gguf, model-v22.gguf, etc.

        Args:
            model_id: HuggingFace repository ID
            repo_type: Type of repo

        Returns:
            List of filenames with the highest version, or None if no versioned files found
        """
        try:
            api = HfApi()
            files = api.list_repo_files(repo_id=model_id, repo_type=repo_type)
        except Exception as e:
            self._log(f"  ⚠ Could not list files for {model_id}: {e}")
            return None

        if not files:
            return None

        # Pattern to find version numbers in filenames
        # Matches: v23, v11.1, v1.2.3, -v5, _v18.1, etc.
        version_in_filename = re.compile(r'[-_]?v?(\d+(?:\.\d+)*)(?=[-_\.]|$)', re.IGNORECASE)

        versioned_files = []

        for filename in files:
            # Skip metadata files
            if filename in ['.gitattributes', 'README.md', 'config.json', 'model_index.json']:
                continue

            match = version_in_filename.search(filename)
            if match:
                try:
                    ver_str = match.group(1)
                    ver = pkg_version.parse(ver_str)
                    versioned_files.append((ver, filename))
                except Exception:
                    continue

        if not versioned_files:
            self._log(f"  ℹ No versioned files found in {model_id}")
            return None

        # Find the highest version
        versioned_files.sort(key=lambda x: x[0], reverse=True)
        highest_version = versioned_files[0][0]

        # Get all files with the highest version
        latest_files = [f[1] for f in versioned_files if f[0] == highest_version]

        self._log(f"  ℹ Found versioned files, highest version: {highest_version}")
        self._log(f"  ℹ Files to download: {latest_files}")

        return latest_files

    def download(self,
                 model_id: str,
                 target_dir: str,
                 ignore_patterns: Optional[List[str]] = None,
                 force_redownload: Optional[bool] = None,
                 create_dir: Optional[bool] = None,
                 label: Optional[str] = None,
                 verbose: Optional[bool] = None) -> bool:
        """
        Download a model from Hugging Face Hub.

        All logic, authentication, and printing handled internally.

        Args:
            model_id: HuggingFace repository ID (e.g., "facebook/mms-tts-rus")
            target_dir: Local directory to download to
            ignore_patterns: File patterns to ignore (uses class default if None)
            force_redownload: Re-download if exists (uses class default if None)
            create_dir: Create target directory (uses class default if None)
            label: Display label (auto-generated from model_id if None)
            verbose: Override class verbose setting for this call

        Returns:
            bool: True if download succeeded, False otherwise
        """
        # Use call-specific verbose or fall back to class setting
        call_verbose = verbose if verbose is not None else self.verbose

        # Apply class-level defaults if not overridden
        if ignore_patterns is None:
            ignore_patterns = self.DEFAULT_IGNORE_PATTERNS
        if force_redownload is None:
            force_redownload = self.DEFAULT_FORCE_REDOWNLOAD
        if create_dir is None:
            create_dir = self.DEFAULT_CREATE_DIR

        # Auto-generate label from model_id if not provided
        if label is None:
            label = self._generate_label_from_model_id(model_id)

        normalized_dir = self.normalize_model_path(target_dir)

        # Helper for conditional logging
        def log(msg):
            if call_verbose:
                print(msg)

        # Authenticate on first download attempt
        self._authenticate()

        # Check for existing download
        existing_size = self._check_existing(normalized_dir, force_redownload)
        if existing_size is not None:
            log(f"✓ {label} already exists ({existing_size:.2f} GB) - skipping")
            return True

        try:
            if create_dir:
                Path(normalized_dir).mkdir(parents=True, exist_ok=True)

            log(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}  Downloading {label} to: {normalized_dir}")
            log(f"  Repo: {model_id}")

            # Strategy 1: Check for git tags first
            if self.try_version:
                best_revision = self._get_best_revision(model_id)
            else:
                best_revision = None

            # Strategy 2: If no tags, check for versioned files in filenames
            allow_patterns = None
            if best_revision is None and self.try_version:
                latest_files = self._get_latest_version_files(model_id)
                if latest_files:
                    allow_patterns = latest_files

            # Prepare kwargs for snapshot_download
            download_kwargs = {
                "repo_id": model_id,
                "local_dir": normalized_dir,
                "ignore_patterns": ignore_patterns,
                "force_download": force_redownload,
                "token": self._get_token()
            }

            # Only add revision if we found a valid git tag
            if best_revision is not None:
                download_kwargs["revision"] = best_revision
                log(f"  Using git revision: {best_revision}")

            # Only add allow_patterns if we found versioned files
            if allow_patterns is not None:
                download_kwargs["allow_patterns"] = allow_patterns
                log(f"  Using latest version files: {allow_patterns}")
            else:
                log(f"  ℹ No version info found - downloading full repo")

            start_time = time.time()

            snapshot_download(**download_kwargs)

            elapsed = time.time() - start_time
            size_gb = self._get_directory_size_gb(normalized_dir)

            log(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} ✓ {label} downloaded successfully ({size_gb:.2f} GB)")
            log(f"  Time elapsed: {elapsed / 60:.1f} minutes")
            return True

        except Exception as e:
            log(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} ✗ {label} download failed: {e}")
            return False

    def download_batch(self,
                       models: List[dict],
                       summary: bool = True,
                       verbose: Optional[bool] = None) -> dict:
        """
        Download multiple models with a summary report.

        Args:
            models: List of dicts with keys:
                - model_id (required)
                - target_dir (required)
                - label, ignore_patterns, force_redownload (all optional)
            summary: Whether to print final summary
            verbose: Override class verbose setting

        Returns:
            dict: {model_id: success_bool} mapping
        """
        call_verbose = verbose if verbose is not None else self.verbose

        def log(msg):
            if call_verbose:
                print(msg)

        if summary:
            log("\n" + "=" * 70)
            log("DOWNLOADING MULTIPLE MODELS")
            log("=" * 70)
            start_time = time.time()

        results = {}

        for i, config in enumerate(models, 1):
            if summary:
                display_label = config.get('label') or self._generate_label_from_model_id(config['model_id'])
                log(f"\n[{i}/{len(models)}] Downloading {display_label}...")
                log("-" * 70)

            success = self.download(**config, verbose=call_verbose)
            results[config['model_id']] = success

        if summary:
            elapsed = time.time() - start_time
            log("\n" + "=" * 70)
            log("DOWNLOAD SUMMARY")
            log("=" * 70)

            for model_id, success in results.items():
                label = self._generate_label_from_model_id(model_id)
                status = "✓ SUCCESS" if success else "✗ FAILED"
                log(f"{label:30s}: {status}")

            log(f"Total Time: {elapsed / 60:.1f} minutes")
            log("=" * 70)

            if all(results.values()):
                log("\n🎉 All models downloaded successfully!")
            else:
                log("\n⚠️  Some downloads failed. Check errors above.")
                log("💡 Tip: Set force_redownload=True to retry failed downloads")

        return results