# export_audio.py
"""
Export audio+text pairs from cached dataset.
Function name: export_audio (consistent, as requested)
"""

import os
import numpy as np
from scipy.io import wavfile
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import io


def export_audio(
        output_dir: str = r"D:\Data\audio_test",
        cache_dir: str = r"D:\Data\audio\kijjjj",
        num_samples: int = 10,
        seed: int = 42
) -> list:
    """
    Export N randomly selected audio+text pairs from cached dataset.
    Selection is repeatable via seed parameter.

    Creates files: sample_0000.wav + sample_0000.txt, etc.

    Parameters
    ----------
    output_dir : str
        Destination folder for exported files
    cache_dir : str
        Where dataset parquet/arrow files are cached
    num_samples : int
        Number of samples to export
    seed : int, optional (default=42)
        Random seed for repeatable sampling

    Returns
    -------
    list of exported file paths
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    print(f"🎲 Random seed: {seed}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find cached files - support both .arrow and .parquet
    arrow_files = sorted(Path(cache_dir).rglob("*.arrow"))
    parquet_files = sorted(Path(cache_dir).rglob("*.parquet"))
    data_files = arrow_files if arrow_files else parquet_files

    if not data_files:
        raise FileNotFoundError(f"No parquet/arrow files found in {cache_dir}")

    print(f"Found {len(data_files)} file(s) in cache")

    # === Phase 1: Collect all available (file, row_idx) pairs ===
    print("📚 Scanning dataset for available samples...")
    all_samples = []  # List of (data_file, row_idx) tuples

    for data_file in data_files:
        # Try multiple methods to read the file
        df = None
        for read_method in [_read_with_pq, _read_with_pa_stream, _read_with_pa_file]:
            try:
                df = read_method(data_file)
                break
            except Exception:
                continue

        if df is None:
            print(f"⚠ Could not read {data_file.name}, skipping...")
            continue

        # Add all row indices from this file
        for idx in range(len(df)):
            all_samples.append((data_file, idx))

    print(f"✅ Found {len(all_samples)} total samples available")

    if num_samples > len(all_samples):
        print(f"⚠ Requested {num_samples} samples but only {len(all_samples)} available")
        num_samples = len(all_samples)

    # === Phase 2: Randomly select samples (with seed) ===
    selected_indices = np.random.choice(
        len(all_samples),
        size=num_samples,
        replace=False
    )
    selected_samples = [all_samples[i] for i in selected_indices]
    print(f"🎯 Selected {len(selected_samples)} samples for export")

    # === Phase 3: Export selected samples ===
    exported_files = []

    for export_idx, (data_file, row_idx) in enumerate(selected_samples):
        name = f"sample_{export_idx:04d}"

        # Re-read the file to get the specific row
        df = None
        for read_method in [_read_with_pq, _read_with_pa_stream, _read_with_pa_file]:
            try:
                df = read_method(data_file)
                break
            except Exception:
                continue

        if df is None:
            print(f"⚠ Could not re-read {data_file.name} for sample {name}, skipping...")
            continue

        row = df.iloc[row_idx]

        # === Save transcription (.txt) ===
        text = str(row['text']).strip()
        txt_path = output_dir / f"{name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        exported_files.append(str(txt_path))

        # === Save audio (.wav) ===
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
                print(f"✓ {name}: {text[:50]}...")

            except Exception:
                sr = audio_val.get('sampling_rate', 16000)
                try:
                    wavfile.write(str(wav_path), sr,
                                  np.frombuffer(audio_bytes, dtype=np.int16))
                    exported_files.append(str(wav_path))
                    print(f"✓ {name} (raw): {text[:50]}...")
                except Exception:
                    bin_path = output_dir / f"{name}.bin"
                    with open(bin_path, 'wb') as f:
                        f.write(audio_bytes)
                    exported_files.append(str(bin_path))
                    print(f"⚠ {name}.bin (raw bytes): {text[:50]}...")
        else:
            print(f"⚠ {name}: No audio data available")

    print(f"\n✅ Exported {len(selected_samples)} samples to {output_dir}")
    print(f"📁 Total files: {len(exported_files)}")
    print(f"🔁 Seed: {seed} (use same seed for identical results)")

    return exported_files


def _read_with_pq(filepath: Path):
    """Try reading with parquet reader."""
    table = pq.read_table(filepath)
    return table.to_pandas()


def _read_with_pa_stream(filepath: Path):
    """Try reading with Arrow stream reader."""
    with pa.memory_map(str(filepath)) as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()
    return table.to_pandas()


def _read_with_pa_file(filepath: Path):
    """Try reading with Arrow file reader."""
    with pa.memory_map(str(filepath)) as source:
        reader = pa.ipc.open_file(source)
        table = reader.read_all()
    return table.to_pandas()


def do_export_10_samples():
    # Export 10 randomly selected samples with repeatable seed
    files = export_audio(
        output_dir=r"D:\Data\audio_test",
        cache_dir=r"D:\Data\audio\kijjjj",
        num_samples=10,
        seed=42  # Change this seed for different random selection
    )

    # Print summary
    print(f"\n📄 Exported files:")
    for f in sorted(set(files)):
        print(f"   • {Path(f).name}")