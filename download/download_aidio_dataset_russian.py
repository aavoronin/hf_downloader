# download_aidio_dataset_russian.py
"""
Download and export audio datasets from Hugging Face.
Function names (FIXED - no changes): download_dataset, download_audio_dataset_russian, export_audio
"""

from datasets import load_dataset
from download.my_token import load_hf_token
import os
import logging
import shutil
import numpy as np
from scipy.io import wavfile
import pyarrow.parquet as pq
from pathlib import Path
import io
from typing import Optional, List
from huggingface_hub import login

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_dataset(
        dest_dir: str,
        dataset_name: str,
        split: str = "train",
        streaming: bool = False,
        hf_token: Optional[str] = None,
        verify_disk_space: bool = True,
        min_space_gb: float = 50.0,
        dataset_type: str = "audio",
        **load_kwargs
) -> object:
    """
    Generic function to download any Hugging Face dataset.
    """
    dest_dir = os.path.normpath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"Destination directory: {dest_dir}")

    if verify_disk_space and not streaming:
        drive = os.path.splitdrive(dest_dir)[0] + "\\" if os.path.splitdrive(dest_dir)[0] else dest_dir
        try:
            total, used, free = shutil.disk_usage(drive)
            free_gb = free / (1024 ** 3)
            logger.info(f"Free space on {drive}: {free_gb:.2f} GB")
            if free_gb < min_space_gb:
                raise ValueError(
                    f"Insufficient disk space. Required: ~{min_space_gb} GB, "
                    f"Available: {free_gb:.2f} GB."
                )
        except OSError as e:
            logger.warning(f"Could not check disk space: {e}. Proceeding.")

    token = hf_token
    if token is None:
        try:
            logger.info("Loading Hugging Face token via load_hf_token()...")
            token = load_hf_token()
            if not token:
                raise ValueError("load_hf_token() returned empty token")
        except ImportError as e:
            raise ImportError("Failed to import load_hf_token from download.my_token.") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load HF token: {e}") from e

    logger.info("Authenticating with Hugging Face...")
    login(token=token)

    try:
        logger.info(f"Loading {dataset_type} dataset '{dataset_name}' (split='{split}', streaming={streaming})...")

        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            cache_dir=dest_dir,
            **load_kwargs
        )

        if not streaming:
            logger.info(f"✓ Dataset loaded successfully!")
            logger.info(f"  • Number of samples: {len(dataset):,}")

            # ✅ FIX: Use column_names only - does NOT trigger torchcodec
            sample_keys = dataset.column_names if len(dataset) > 0 else []
            logger.info(f"  • Column names: {sample_keys}")

            # ✅ FIX: Removed dataset[0] access that triggered torchcodec import
            # Audio metadata logging removed to avoid ImportError

        else:
            logger.info("✓ Dataset loaded in streaming mode.")

        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


def download_audio_dataset_russian(
        dest_dir: str = r"D:\Data\audio\kijjjj",
        dataset_name: str = "kijjjj/audio_data_russian",
        split: str = "train",
        streaming: bool = False,
        hf_token: str = None,
        verify_disk_space: bool = True,
        min_space_gb: float = 120.0,
        **load_kwargs
) -> object:
    """
    Wrapper for downloading kijjjj/audio_data_russian.
    """
    return download_dataset(
        dest_dir=dest_dir,
        dataset_name=dataset_name,
        split=split,
        streaming=streaming,
        hf_token=hf_token,
        verify_disk_space=verify_disk_space,
        min_space_gb=min_space_gb,
        dataset_type="audio",
        **load_kwargs
    )


def export_audio(
        output_dir: str = r"D:\Data\audio_test",
        cache_dir: str = r"D:\Data\audio\kijjjj",
        num_samples: int = 10,
        dataset_name: str = "kijjjj/audio_data_russian",
        split: str = "train"
) -> List[str]:
    """
    Export N audio+text pairs as WAV+TXT files.
    Function name: export_audio (unchanged, as requested)
    """
    # Ensure dataset is cached (non-streaming to get parquet files)
    logger.info(f"Ensuring dataset '{dataset_name}' is cached...")
    _ = download_dataset(
        dest_dir=cache_dir,
        dataset_name=dataset_name,
        split=split,
        streaming=False
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find parquet files in cache
    dataset_cache_name = dataset_name.replace("/", "___")
    parquet_search_path = Path(cache_dir) / dataset_cache_name
    parquet_files = sorted(parquet_search_path.rglob("*.parquet"))

    if not parquet_files:
        parquet_files = sorted(Path(cache_dir).rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {cache_dir}")

    logger.info(f"Found {len(parquet_files)} parquet file(s)")

    exported_files = []
    exported_count = 0

    for pq_file in parquet_files:
        if exported_count >= num_samples:
            break
        logger.info(f"Reading {pq_file.name}...")
        table = pq.read_table(pq_file)
        df = table.to_pandas()

        for idx in range(len(df)):
            if exported_count >= num_samples:
                break

            name = f"sample_{exported_count:04d}"
            row = df.iloc[idx]

            # Save text
            text = str(row['text']).strip()
            txt_path = output_dir / f"{name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            exported_files.append(str(txt_path))

            # Save audio
            audio_val = row['audio']
            wav_path = output_dir / f"{name}.wav"

            if isinstance(audio_val, dict) and 'bytes' in audio_val:
                audio_bytes = audio_val['bytes']
                try:
                    wav_io = io.BytesIO(audio_bytes)
                    sr, data = wavfile.read(wav_io)
                    if data.dtype in [np.float32, np.float64]:
                        data = np.clip(data, -1.0, 1.0)
                        data = (data * 32767).astype(np.int16)
                    if data.ndim > 1:
                        data = data[:, 0]
                    wavfile.write(str(wav_path), sr, data)
                    exported_files.append(str(wav_path))
                    logger.info(f"✓ {name}: {text[:50]}...")
                except Exception:
                    sr = audio_val.get('sampling_rate', 16000)
                    try:
                        wavfile.write(str(wav_path), sr, np.frombuffer(audio_bytes, dtype=np.int16))
                        exported_files.append(str(wav_path))
                        logger.info(f"✓ {name} (raw): {text[:50]}...")
                    except Exception:
                        bin_path = output_dir / f"{name}.bin"
                        with open(bin_path, 'wb') as f:
                            f.write(audio_bytes)
                        exported_files.append(str(bin_path))
                        logger.warning(f"⚠ {name}.bin (raw): {text[:50]}...")
            else:
                logger.warning(f"⚠ {name}: No audio data")

            exported_count += 1

    logger.info(f"✅ Exported {exported_count} samples to {output_dir}")
    return exported_files