# download_aidio_dataset_russian.py
"""
Download and export audio datasets from Hugging Face.
Function names (FIXED - no changes): download_dataset, download_audio_dataset_russian, export_audio
"""

from datasets import load_dataset, Features, Value, Audio
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
import requests

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
            sample_keys = dataset.column_names if len(dataset) > 0 else []
            logger.info(f"  • Column names: {sample_keys}")
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
    logger.info(f"Ensuring dataset '{dataset_name}' is cached...")
    _ = download_dataset(
        dest_dir=cache_dir,
        dataset_name=dataset_name,
        split=split,
        streaming=False
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

            text = str(row['text']).strip()
            txt_path = output_dir / f"{name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            exported_files.append(str(txt_path))

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


# =============================================================================
# ENGLISH DATASET: LIBRISPEECH ASR (2 simple methods)
# WORKS WITHOUT torchcodec: uses Audio(decode=False) + HTTP download
# =============================================================================

def download_librispeech_english(
        dest_dir: str = r"D:\Data\audio\librispeech_en",
        dataset_name: str = "openslr/librispeech_asr",
        config: str = "clean",
        split: str = "train.100",
        streaming: bool = True,
        hf_token: str = None,
        verify_disk_space: bool = False,
        min_space_gb: float = 1.0,
        **load_kwargs
) -> object:
    """
    Download LibriSpeech ASR dataset (English) in streaming mode.

    ✅ Uses Audio(decode=False) to avoid torchcodec/FFmpeg dependency.
    ✅ Features schema matches actual dataset columns.
    """
    dest_dir = os.path.normpath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"Destination directory: {dest_dir}")

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
        logger.info(
            f"Loading audio dataset '{dataset_name}' (config='{config}', split='{split}', streaming={streaming})...")

        # ✅ FIX: Exact schema matching openslr/librispeech_asr
        # Actual columns: file, audio{bytes,path}, text, speaker_id, chapter_id, id
        features = Features({
            "file": Value("string"),  # ✅ ADD THIS - was missing!
            "audio": Audio(decode=False),  # ✅ No decoding → no torchcodec
            "text": Value("string"),
            "speaker_id": Value("int64"),
            "chapter_id": Value("int64"),
            "id": Value("string")
        })

        dataset = load_dataset(
            dataset_name,
            config,
            split=split,
            streaming=streaming,
            features=features,
            **load_kwargs
        )

        logger.info("✓ Dataset loaded in streaming mode (audio as bytes/path)")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


def _download_wav_bytes(url: str, headers: Optional[dict] = None, timeout: int = 30) -> Optional[bytes]:
    """Download raw WAV bytes from URL."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return None


def export_librispeech_samples(
        output_dir: str = r"D:\Data\audio_test\librispeech",
        cache_dir: str = r"D:\Data\audio\librispeech_en",
        num_samples: int = 10,
        dataset_name: str = "openslr/librispeech_asr",
        config: str = "clean",
        split: str = "train.100",
        max_phrase_words: Optional[int] = 10,
        max_duration_sec: Optional[float] = 15.0,
        hf_token: Optional[str] = None
) -> List[str]:
    """
    Extract N audio+text pairs from LibriSpeech as separate WAV+TXT files.

    ✅ Downloads audio via HTTP requests - NO torchcodec or FFmpeg required.
    """
    logger.info(f"Loading LibriSpeech dataset in streaming mode...")
    dataset = download_librispeech_english(
        dest_dir=cache_dir,
        dataset_name=dataset_name,
        config=config,
        split=split,
        streaming=True
    )

    # Prepare auth header if token provided
    headers = {}
    if hf_token is None:
        try:
            hf_token = load_hf_token()
        except:
            pass
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    # Apply optional filters for short phrases (text-only, before download)
    if max_phrase_words is not None:
        logger.info(f"Filtering by text: max_words={max_phrase_words}")

        def filter_short_text(example):
            text = example.get("text") or ""
            return len(str(text).split()) <= max_phrase_words

        dataset = dataset.filter(filter_short_text)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []
    exported_count = 0
    skipped_count = 0

    for sample in dataset:
        if exported_count >= num_samples:
            break

        try:
            name = f"librispeech_{exported_count:04d}"

            # --- Save text ---
            text = str(sample.get("text") or "").strip()
            if not text:
                continue
            txt_path = output_dir / f"{name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # --- Download and save audio via HTTP ---
            audio_val = sample.get("audio", {})
            wav_path = output_dir / f"{name}.wav"

            # Audio(decode=False) returns: {'bytes': ..., 'path': ...}
            audio_bytes = audio_val.get("bytes") if isinstance(audio_val, dict) else None
            audio_url = audio_val.get("path") if isinstance(audio_val, dict) else None

            if audio_bytes:
                wav_bytes = audio_bytes
            elif audio_url:
                logger.info(f"Downloading: {audio_url[:80]}...")
                wav_bytes = _download_wav_bytes(audio_url, headers=headers)
                if wav_bytes is None:
                    logger.warning(f"⚠ {name}: Failed to download, skipping")
                    if txt_path.exists():
                        txt_path.unlink()
                    skipped_count += 1
                    continue
            else:
                logger.warning(f"⚠ {name}: No audio path or bytes available")
                if txt_path.exists():
                    txt_path.unlink()
                continue

            # Write WAV file
            with open(wav_path, 'wb') as f:
                f.write(wav_bytes)

            # Get duration for logging/filtering
            duration = None
            try:
                wav_io = io.BytesIO(wav_bytes)
                sr, data = wavfile.read(wav_io)
                duration = len(data) / sr if sr > 0 else 0
                if max_duration_sec is not None and duration > max_duration_sec:
                    logger.info(f"⊘ {name}: Too long ({duration:.2f}s), skipping")
                    txt_path.unlink()
                    wav_path.unlink()
                    skipped_count += 1
                    continue
            except Exception as e:
                logger.debug(f"Could not read WAV header for {name}: {e}")

            exported_files.append(str(txt_path))
            exported_files.append(str(wav_path))

            duration_str = f" ({duration:.2f}s)" if duration else ""
            logger.info(f"✓ {name}: {text[:50]}...{duration_str}")
            exported_count += 1

        except Exception as e:
            logger.error(f"✗ Error exporting sample {exported_count}: {e}")
            continue

    logger.info(f"✅ Exported {exported_count}/{num_samples} LibriSpeech samples to {output_dir}")
    if skipped_count > 0:
        logger.info(f"   (Skipped {skipped_count} samples)")
    return exported_files



    print(f"\n✅ Exported {len(files) // 2} pairs:")
    for i in range(0, len(files), 2):
        print(f"  • {files[i]} + {files[i + 1]}")