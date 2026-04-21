# ==========# ASRManager.py
"""
ASR Model Manager - Fixed for CUDA 12.8 + sm_120 (RTX 5070 Ti)
Expanded with run_test2 for batch processing and aggregated statistics.
"""
import os
import time
import traceback
import re
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import sys
from io import StringIO
from ASR.ASRModelFactory import ASRModelFactory
from ASR.AutomaticSpeechRecognition import AutomaticSpeechRecognition
from ASR.ModelInfo import ModelInfo
from ASR.ProcessingResult import ProcessingResult
# === Direct imports (no try/except) ===
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
from moviepy.audio.io.AudioFileClip import AudioFileClip
from jiwer import wer
from text2digits import text2digits

class ASRManager:
    def __init__(self, root_folder: str):
        self.factory = ASRModelFactory(root_folder)
        self.t2d = text2digits.Text2Digits()
        self.model_filter_mode = 'all'  # Options: 'all', 'ru', 'en'

    def set_model_filter_mode(self, mode: str):
        """Switch between model sets: 'all', 'ru_models', or 'en_models'."""
        mode = mode.lower().replace('_models', '')
        if mode in ('all', 'ru', 'en'):
            self.model_filter_mode = mode
        else:
            raise ValueError("mode must be 'all', 'ru_models', or 'en_models'")

    def list_models(self) -> List[ModelInfo]:
        return self.factory.list_available_models()

    def get_model(self, model_name: str) -> Optional[AutomaticSpeechRecognition]:
        return self.factory.create(model_name)

    @staticmethod
    def _clean_reference_text(text: str) -> str:
        """Clean reference text: normalize whitespace, remove newlines/carriage returns."""
        text = text.strip()
        text = text.replace("\r", " ").replace("\n", " ")
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

    def _get_audio_duration(self, audio_path: str) -> float:
        clip = AudioFileClip(audio_path)
        duration = clip.duration
        clip.close()
        return duration

    @staticmethod
    def _normalize_text_for_asr(text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\dа-яё]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _convert_numbers_to_digits(self, text: str) -> str:
        return self.t2d.convert(text)

    def calculate_asr_metrics(
        self,
        reference: str,
        hypothesis: str
    ) -> Dict[str, Any]:
        ref_norm = self._normalize_text_for_asr(reference)
        hyp_norm = self._normalize_text_for_asr(hypothesis)
        ref_digits = self._convert_numbers_to_digits(ref_norm)
        hyp_digits = self._convert_numbers_to_digits(hyp_norm)
        similarity = max(0.0, 1.0 - wer(ref_digits, hyp_digits))
        return {
            'reference_normalized': ref_digits,
            'hypothesis_normalized': hyp_digits,
            'similarity': similarity,
        }

    def _normalize_for_display(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'\b(plus|\+)\b', '+', text)
        text = self.t2d.convert(text)
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def run_test2(
        self,
        test_cases: List[Dict[str, str]],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        total_start = time.time()
        if not test_cases:
            print("❌ No test cases provided.")
            return {}

        # ==========================================================
        # DYNAMIC MODEL SELECTION BASED ON FILTER SWITCH
        # ==========================================================
        if model_names is None:
            mode = self.model_filter_mode
            if mode == 'all':
                available = self.factory.list_available_models()
                model_names = [m.name for m in available]
            else:
                lang = 'ru' if mode == 'ru' else 'en'
                factory_list = ASRModelFactory.GetModelsList(lang)
                available = self.factory.list_available_models()
                available_names = {m.name for m in available}
                model_names = [m["model_name"] for m in factory_list if m["model_name"] in available_names]
        # ==========================================================

        print(f"\n🚀 Starting Batch Test (run_test2) - Model-First Loop")
        print(f"📂 Total Files: {len(test_cases)}")
        print(f"🤖 Total Models: {len(model_names)}")
        print("-" * 60)

        reference_cache = {}
        for idx, case in enumerate(test_cases):
            ref_path = case.get('reference')
            if ref_path and ref_path not in reference_cache:
                ref_text = self._load_reference_text(ref_path)
                if ref_text is not None:
                    reference_cache[ref_path] = ref_text
                else:
                    print(f"   ⚠️  Could not load reference: {ref_path}")

        valid_cases = [
            case for case in test_cases
            if case.get('reference') in reference_cache and os.path.exists(case.get('audio', ''))
        ]
        if not valid_cases:
            print("❌ No valid test cases after filtering.")
            return {}
        print(f"✅ Loaded {len(valid_cases)} valid test cases")

        file_names = [Path(case.get('audio', '')).name for case in valid_cases]
        model_stats = {
            name: {'similarities': {}, 'times': {}, 'rtfs': [], 'successes': 0}
            for name in model_names
        }
        detailed_results = []
        file_results = {case.get('audio'): {'reference': None, 'predictions': []} for case in valid_cases}

        for model_idx, model_name in enumerate(model_names, 1):
            print(f"\n{'=' * 70}")
            print(f"🤖 Model [{model_idx}/{len(model_names)}]: {model_name}")
            print(f"{'=' * 70}")
            if self.factory.is_model_faulty(model_name):
                print(f"⊘ SKIPPED: Model marked as faulty (>10 errors)")
                continue
            start_init = time.time()
            model = self.factory.create(model_name)
            if model is None:
                print("Model initialization returned None")
                continue
            init_time = time.time() - start_init
            print(f"{datetime.now().isoformat(timespec='milliseconds')} Model loaded in {init_time:.2f}s")

            for case_idx, case in enumerate(valid_cases, 1):
                audio_path = case.get('audio')
                reference_path = case.get('reference')
                reference_text = reference_cache.get(reference_path)
                file_name = Path(audio_path).name
                if not reference_text:
                    print(f"   ⚠️  [{case_idx}/{len(valid_cases)}] Skipping: no reference text")
                    continue
                print(f"   📁 [{case_idx}/{len(valid_cases)}] {file_name}", end=" ... ")
                start_infer = time.time()
                success = False
                similarity = 0.0
                elapsed = 0.0
                predicted_text = ""
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    predicted_text = model.process(audio_path)
                except Exception as e:
                    print(e)
                    continue
                finally:
                    sys.stdout = old_stdout
                elapsed = time.time() - start_infer
                metrics = self.calculate_asr_metrics(reference_text, predicted_text)
                similarity = metrics['similarity']
                success = True
                model_stats[model_name]['successes'] += 1
                model_stats[model_name]['similarities'][file_name] = similarity
                model_stats[model_name]['times'][file_name] = elapsed
                rtf = 0.0
                duration = self._get_audio_duration(audio_path)
                if duration > 0:
                    rtf = elapsed / duration
                model_stats[model_name]['rtfs'].append(rtf)

                if file_results[audio_path]['reference'] is None:
                    file_results[audio_path]['reference'] = self._normalize_for_display(reference_text)
                file_results[audio_path]['predictions'].append({
                    'model': model_name,
                    'predicted_normalized': self._normalize_for_display(predicted_text),
                    'similarity': similarity,
                    'time': elapsed,
                    'rtf': rtf
                })
                status_icon = "✓" if similarity > 0.5 else "⚠"
                print(f"{status_icon} Sim={similarity:.4f}")
                detailed_results.append({
                    'file': file_name,
                    'model': model_name,
                    'similarity': similarity,
                    'time': elapsed,
                    'rtf': rtf,
                    'success': success,
                    'predicted': predicted_text[:100] + "..." if len(predicted_text) > 100 else predicted_text
                })

        print(f"\n{'=' * 80}")
        print(f"📄 RESULTS BY FILE (Expected + All Models)")
        print(f"{'=' * 80}")
        for audio_path, data in file_results.items():
            file_name = Path(audio_path).name
            print(f"\n📁 File: {file_name}")
            print(f"   🟢 Expected: {data['reference']}")
            for pred in data['predictions']:
                print(f"      {pred['predicted_normalized']} --- {pred['model']}")

        print(f"\n{'=' * 80}")
        print(f"📊 COMBINED STATISTICS (All Files)")
        print(f"{'=' * 80}")
        summary_rows = []
        for model_name in model_names:
            stats = model_stats[model_name]
            if stats['successes'] == 0:
                continue
            similarities = list(stats['similarities'].values())
            times = list(stats['times'].values())
            rtfs = stats['rtfs']
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            min_time = min(times) if times else 0.0
            max_time = max(times) if times else 0.0
            avg_t = sum(rtfs) / len(rtfs) if rtfs else 0.0
            sim_70 = sum(1 for s in similarities if s > 0.7)
            sim_80 = sum(1 for s in similarities if s > 0.8)
            sim_90 = sum(1 for s in similarities if s > 0.9)
            sim_95 = sum(1 for s in similarities if s > 0.95)
            sim_97 = sum(1 for s in similarities if s > 0.97)
            sim_98 = sum(1 for s in similarities if s > 0.98)
            sim_99 = sum(1 for s in similarities if s > 0.99)
            summary_rows.append({
                'model_name': model_name,
                'avg_sim': avg_sim,
                'avg_t': avg_t,
                'sim_70': sim_70,
                'sim_80': sim_80,
                'sim_90': sim_90,
                'sim_95': sim_95,
                'sim_97': sim_97,
                'sim_98': sim_98,
                'sim_99': sim_99,
                'min_time': min_time,
                'max_time': max_time,
                'file_similarities': stats['similarities'],
                'successes': stats['successes']
            })
        summary_rows.sort(key=lambda x: x['avg_sim'], reverse=True)
        header = f"{'Model Name':<45} {'AvgSim':>7} {'Avg_t':>7} {'>0.7':>5} {'>0.8':>5} {'>0.9':>5} {'>0.95':>5} {'>0.97':>5} {'>0.98':>5} {'>0.99':>5} {'MinT':>6} {'MaxT':>6}"
        for fn in file_names:
            display_name = fn[:10] + ".." if len(fn) > 12 else fn
            header += f" {display_name:>12}"
        print(header)
        print("-" * len(header))
        for row in summary_rows:
            line = f"{row['model_name']:<45} {row['avg_sim']:>7.4f} {row['avg_t']:>7.2f} {row['sim_70']:>5} {row['sim_80']:>5} {row['sim_90']:>5} {row['sim_95']:>5} {row['sim_97']:>5} {row['sim_98']:>5} {row['sim_99']:>5} {row['min_time']:>6.2f} {row['max_time']:>6.2f}"
            for fn in file_names:
                sim = row['file_similarities'].get(fn, 0.0)
                line += f" {sim:>12.4f}"
            print(line)

        stats_payload = {
            'test_type': 'batch_run_test2_model_first',
            'total_files': len(valid_cases),
            'file_names': file_names,
            'models_tested': len(model_names),
            'model_summary': summary_rows,
            'detailed_results': detailed_results
        }
        self.factory.save_statistics(stats_payload)
        print(f"\n💾 Statistics saved to {self.factory.stats_path}")
        total_elapsed = time.time() - total_start
        print(f"\n⏱ Total Run Time: {total_elapsed:.2f} seconds")
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