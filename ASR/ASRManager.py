# ==========# ASRManager.py
"""
ASR Model Manager - Fixed for CUDA 12.8 + sm_120 (RTX 5070 Ti)
Expanded with run_test2 for batch processing and aggregated statistics.
"""

import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional, Dict, Any

from ASR.ASRModelFactory import ASRModelFactory
from ASR.AutomaticSpeechRecognition import AutomaticSpeechRecognition
from ASR.ModelInfo import ModelInfo
from ASR.ProcessingResult import ProcessingResult

# === Import each package independently ===
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ Warning: torch not available")

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠ Warning: transformers import failed: {e}")

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False
    print("⚠ Warning: librosa not available")

try:
    import moviepy.editor as mp

    MOVIEPY_AVAILABLE = True
except ImportError:
    moviepy = None
    MOVIEPY_AVAILABLE = False
    print("⚠ Warning: moviepy not available")


class ASRManager:

    def __init__(self, root_folder: str):
        self.factory = ASRModelFactory(root_folder)

    def list_models(self) -> List[ModelInfo]:
        return self.factory.list_available_models()

    def get_model(self, model_name: str) -> Optional[AutomaticSpeechRecognition]:
        return self.factory.create(model_name)

    @staticmethod
    def _clean_reference_text(text: str) -> str:
        """
        Clean reference text: normalize whitespace, remove newlines/carriage returns.
        Reusable method for both run_test and run_test2.
        """
        text = text.strip()
        text = text.replace("\r", " ").replace("\n", " ")
        # Collapse multiple spaces into single space (iterate to handle edge cases)
        while "  " in text:
            text = text.replace("  ", " ")
        return text

    def _load_reference_text(self, reference_path: str) -> Optional[str]:
        """Load and clean reference text from file."""
        try:
            with open(reference_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                return self._clean_reference_text(raw_text)
        except FileNotFoundError:
            print(f"❌ Reference file not found: {reference_path}")
            return None
        except Exception as e:
            print(f"❌ Failed to load reference {reference_path}: {e}")
            return None

    def apply_all(
            self,
            input_paths: Union[str, Path, List[Union[str, Path]]],
            model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if model_names is None:
            available = self.factory.list_available_models()
            model_names = [m.name for m in available]
        results = []
        successful_models = []
        for model_name in model_names:
            if self.factory.is_model_faulty(model_name):
                results.append(ProcessingResult(
                    model_name=model_name,
                    text=None,
                    success=False,
                    time_taken=0,
                    datetime_completed=datetime.now().isoformat(),
                    error_message="Model marked as faulty (>10 errors)"
                ))
                continue
            start_time = time.time()
            try:
                model = self.factory.create(model_name)
                if model is None:
                    raise RuntimeError("Failed to create model instance")
                text = model.process(input_paths)
                elapsed = time.time() - start_time
                result = ProcessingResult(
                    model_name=model_name,
                    text=text,
                    success=True,
                    time_taken=elapsed,
                    datetime_completed=datetime.now().isoformat()
                )
                successful_models.append(model_name)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ Model {model_name} failed:")
                print(f"   Error: {str(e)}")
                traceback.print_exc()
                self.factory._log_error(model_name, str(e))
                result = ProcessingResult(
                    model_name=model_name,
                    text=None,
                    success=False,
                    time_taken=elapsed,
                    datetime_completed=datetime.now().isoformat(),
                    error_message=str(e)
                )
            results.append(result)
        stats = {
            'input_paths': [str(p) for p in (input_paths if isinstance(input_paths, list) else [input_paths])],
            'models_tested': len(model_names),
            'successful_models': successful_models,
            'results': [r.to_dict() for r in results]
        }
        self.factory.save_statistics(stats)
        return stats

    def run_test(
            self,
            audio_path: str,
            reference_path: str,
            model_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        # Load and clean reference text using extracted method
        reference_text = self._load_reference_text(reference_path)
        if reference_text is None:
            return []

        print(f"\n🧪 Running ASR Benchmark Test")
        print(f"📁 Audio: {audio_path}")
        print(f"📄 Reference: {reference_path}")
        print(f"📏 Reference length: {len(reference_text)} chars\n")

        if model_names is None:
            available = self.factory.list_available_models()
            model_names = [m.name for m in available]
        print(f"🔄 Testing {len(model_names)} models...\n")
        results = []
        for model_name in model_names:
            if self.factory.is_model_faulty(model_name):
                print(f"⊘ {model_name}: SKIPPED (faulty)")
                continue
            start_time = time.time()
            try:
                print(f"\n{'=' * 60}")
                print(f" Testing: {model_name}")
                print(f"{'=' * 60}")
                model = self.factory.create(model_name)
                if model is None:
                    print(f"✗ {model_name}: FAILED to initialize")
                    continue
                predicted_text = model.process(audio_path)
                print(f"\n📝 Predicted: {predicted_text}")
                elapsed = time.time() - start_time
                similarity = self._normalized_levenshtein_similarity(
                    reference_text.lower(), predicted_text.lower())
                status = "✓" if similarity > 0.5 else "⚠"
                print(f"\n{status} {model_name}")
                print(f"  Similarity: {similarity:.4f} | Time: {elapsed:.2f}s")
                results.append({
                    'model_name': model_name,
                    'similarity': similarity,
                    'success': True,
                    'time_taken': elapsed
                })
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"\n✗ {model_name}: ERROR")
                print(f"  Error: {str(e)}")
                print(f"  Traceback:")
                traceback.print_exc()
                self.factory._log_error(model_name, str(e))
                results.append({
                    'model_name': model_name,
                    'similarity': 0.0,
                    'success': False,
                    'time_taken': elapsed,
                    'error': str(e)
                })
        print(f"\n📊 Test Summary")
        print(f"{'Model Name':<50} {'Similarity':>10} {'Status':>10} {'Time (s)':>10}")
        print("-" * 82)
        for r in sorted(results, key=lambda x: x['similarity'], reverse=True):
            status = "SUCCESS" if r['success'] else "FAILED"
            print(f"{r['model_name']:<50} {r['similarity']:>10.4f} {status:>10} {r['time_taken']:>10.2f}")
        return results

    def run_test2(
            self,
            test_cases: List[Dict[str, str]],
            model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs batch testing on multiple files with MODEL-FIRST loop.
        Initializes each model once, then runs all test cases against it.
        Prints combined statistics for all files against each model.

        Args:
            test_cases: List of dicts, each containing {'audio': path, 'reference': path}
            model_names: Optional list of model names to test. If None, uses all available.
        """
        if not test_cases:
            print("❌ No test cases provided.")
            return {}

        if model_names is None:
            available = self.factory.list_available_models()
            model_names = [m.name for m in available]

        print(f"\n🚀 Starting Batch Test (run_test2) - Model-First Loop")
        print(f"📂 Total Files: {len(test_cases)}")
        print(f"🤖 Total Models: {len(model_names)}")
        print("-" * 60)

        # Pre-load and clean all reference texts once
        print(f"\n📚 Pre-loading reference texts...")
        reference_cache = {}
        for idx, case in enumerate(test_cases):
            ref_path = case.get('reference')
            if ref_path and ref_path not in reference_cache:
                ref_text = self._load_reference_text(ref_path)
                if ref_text is not None:
                    reference_cache[ref_path] = ref_text
                else:
                    print(f"   ⚠️  Could not load reference: {ref_path}")

        # Filter out test cases with missing references
        valid_cases = [
            case for case in test_cases
            if case.get('reference') in reference_cache and os.path.exists(case.get('audio', ''))
        ]

        if not valid_cases:
            print("❌ No valid test cases after filtering.")
            return {}

        print(f"✅ Loaded {len(valid_cases)} valid test cases")

        # Structure to hold aggregated stats per model
        model_stats = {name: {'similarities': [], 'times': [], 'successes': 0, 'total': 0} for name in model_names}
        detailed_results = []

        # === MODEL-FIRST LOOP ===
        for model_idx, model_name in enumerate(model_names, 1):
            print(f"\n{'=' * 70}")
            print(f"🤖 Model [{model_idx}/{len(model_names)}]: {model_name}")
            print(f"{'=' * 70}")

            # Skip faulty models
            if self.factory.is_model_faulty(model_name):
                print(f"⊘ SKIPPED: Model marked as faulty (>10 errors)")
                # Record skipped runs for all cases
                for _ in valid_cases:
                    model_stats[model_name]['total'] += 1
                continue

            # Initialize model ONCE for all test cases
            start_init = time.time()
            try:
                model = self.factory.create(model_name)
                if model is None:
                    raise RuntimeError("Model initialization returned None")
                init_time = time.time() - start_init
                print(f"✅ Model loaded in {init_time:.2f}s")
            except Exception as e:
                init_time = time.time() - start_init
                print(f"✗ Failed to initialize model: {str(e)}")
                self.factory._log_error(model_name, str(e))
                # Record failures for all cases
                for _ in valid_cases:
                    model_stats[model_name]['total'] += 1
                    model_stats[model_name]['times'].append(init_time)
                continue

            # Process ALL test cases with this initialized model
            for case_idx, case in enumerate(valid_cases, 1):
                audio_path = case.get('audio')
                reference_path = case.get('reference')
                reference_text = reference_cache.get(reference_path)

                if not reference_text:
                    print(f"   ⚠️  [{case_idx}/{len(valid_cases)}] Skipping: no reference text")
                    model_stats[model_name]['total'] += 1
                    continue

                print(f"   📁 [{case_idx}/{len(valid_cases)}] {Path(audio_path).name}", end=" ... ")

                start_infer = time.time()
                success = False
                similarity = 0.0

                try:
                    predicted_text = model.process(audio_path)
                    elapsed = time.time() - start_infer

                    similarity = self._normalized_levenshtein_similarity(
                        reference_text.lower(), predicted_text.lower()
                    )
                    success = True
                    model_stats[model_name]['successes'] += 1
                    model_stats[model_name]['similarities'].append(similarity)
                    model_stats[model_name]['times'].append(elapsed)

                    status_icon = "✓" if similarity > 0.5 else "⚠"
                    print(predicted_text)
                    print(f"{status_icon} Sim={similarity:.4f}, Time={elapsed:.2f}s")

                    detailed_results.append({
                        'file': Path(audio_path).name,
                        'model': model_name,
                        'similarity': similarity,
                        'time': elapsed,
                        'success': success,
                        'predicted': predicted_text[:100] + "..." if len(predicted_text) > 100 else predicted_text
                    })

                except Exception as e:
                    elapsed = time.time() - start_infer
                    print(f"✗ ERROR: {str(e)[:40]}")
                    self.factory._log_error(model_name, str(e))
                    model_stats[model_name]['times'].append(elapsed)
                    detailed_results.append({
                        'file': Path(audio_path).name,
                        'model': model_name,
                        'similarity': 0.0,
                        'time': elapsed,
                        'success': False,
                        'error': str(e)
                    })

                model_stats[model_name]['total'] += 1

        # === Print Combined Statistics ===
        print(f"\n{'=' * 80}")
        print(f"📊 COMBINED STATISTICS (All Files)")
        print(f"{'=' * 80}")
        print(f"{'Model Name':<40} {'Avg Sim':>10} {'Best Sim':>10} {'Avg Time':>10} {'Success Rate':>12}")
        print("-" * 80)

        summary_table = []
        for model_name in model_names:
            stats = model_stats[model_name]
            total = stats['total']
            successes = stats['successes']

            if total == 0:
                continue

            avg_sim = sum(stats['similarities']) / len(stats['similarities']) if stats['similarities'] else 0.0
            best_sim = max(stats['similarities']) if stats['similarities'] else 0.0
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0.0
            success_rate = (successes / total) * 100 if total > 0 else 0.0

            summary_table.append({
                'model_name': model_name,
                'avg_sim': avg_sim,
                'best_sim': best_sim,
                'avg_time': avg_time,
                'success_rate': success_rate,
                'total_runs': total
            })

            print(f"{model_name:<40} {avg_sim:>10.4f} {best_sim:>10.4f} {avg_time:>10.2f}s {success_rate:>11.1f}%")

        # Save Statistics
        stats_payload = {
            'test_type': 'batch_run_test2_model_first',
            'total_files': len(valid_cases),
            'models_tested': len(model_names),
            'model_summary': summary_table,
            'detailed_results': detailed_results
        }
        self.factory.save_statistics(stats_payload)
        print(f"\n💾 Statistics saved to {self.factory.stats_path}")

        return stats_payload

    @staticmethod
    def _normalized_levenshtein_similarity(s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        def levenshtein_distance(a: str, b: str) -> int:
            if len(a) < len(b):
                a, b = b, a
            previous_row = list(range(len(b) + 1))
            for i, c1 in enumerate(a):
                current_row = [i + 1]
                for j, c2 in enumerate(b):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        distance = levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)


