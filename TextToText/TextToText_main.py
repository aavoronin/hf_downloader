import time
from TextToText.TestCasesLoader import TestCasesLoaded
from TextToText.TextToTextModelFactory import TextToTextModelFactory
from TextToText.TextToTextModelInfo import TextToTextModelInfo
from TextToText.OracleConverterHelper import OracleConverterHelper

ALLOWED_MODELS = [
    # "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    # "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    # "Qwen/Qwen2.5-Coder-7B-Instruct",
    # "Qwen/Qwen2.5-Coder-14B-Instruct",
    # "PipableAI/pip-sql-1.3b",
    # "prem-research/prem-1B-SQL",
    # "PipableAI/pip-sql-1.3b",
]


class CodeConverter:
    def __init__(self, manager: TextToTextModelFactory, model_info: TextToTextModelInfo,
                 test_cases_loader: TestCasesLoaded):
        self.manager = manager
        self.model_info = model_info
        self.model_name = model_info.name
        self.test_cases_loader = test_cases_loader
        self.test_prompts = test_cases_loader.get_test_prompts()
        self.num_cases = len(self.test_prompts)
        self.model = None

    def _init_model(self) -> bool:
        print(f"\n{'=' * 60}")
        print(f" Testing: {self.model_name}")
        print(f"{'=' * 60}")
        self.model = self.manager.create(self.model_name)
        if self.model is None:
            print(f"✗ {self.model_name}: FAILED to initialize")
            self.manager._log_error(self.model_name, "Initialization returned None")
            return False
        return True

    def _process_case(self, i: int, prompt: str) -> tuple:
        case_start = time.time()
        print(f"\n--- Test Case {i} ---")
        try:
            predicted_text = self.model.process(prompt)
            elapsed_case = time.time() - case_start
            case_success = 1 if predicted_text and len(predicted_text.strip()) > 0 else 0

            output_preview = predicted_text[:2000] if predicted_text else ""
            print(f"Result Case {i}:\n{output_preview}")
            print(f"Time: {elapsed_case:.2f}s | Status: {case_success}")

            prompt_len = len(prompt)
            input_script_len = prompt_len - len(
                self.test_cases_loader.prompt_content) if self.test_cases_loader.prompt_content else prompt_len
            max_tokens = "Unknown"
            if hasattr(self.model, '_custom_config') and self.model._custom_config:
                max_tokens = str(self.model._custom_config.get('max_input_tokens', '?'))

            self.test_cases_loader.save_test_case_result(
                case_index=i - 1,
                success=bool(case_success),
                output_text=predicted_text,
                time_taken=elapsed_case,
                prompt_text=prompt,
                input_script_len=input_script_len,
                output_script_len=len(predicted_text) if predicted_text else 0,
                model_max_tokens=max_tokens,
                model_name=self.model_name
            )
            return case_success, predicted_text
        except Exception as e:
            print(f"Failed Case {i}: {str(e)}")
            elapsed_case = time.time() - case_start
            prompt_len = len(prompt)
            input_script_len = prompt_len - len(
                self.test_cases_loader.prompt_content) if self.test_cases_loader.prompt_content else prompt_len
            max_tokens = "Unknown"
            if hasattr(self.model, '_custom_config') and self.model._custom_config:
                max_tokens = str(self.model._custom_config.get('max_input_tokens', '?'))

            self.test_cases_loader.save_test_case_result(
                case_index=i - 1,
                success=False,
                output_text="",
                time_taken=elapsed_case,
                error_msg=str(e),
                prompt_text=prompt,
                input_script_len=input_script_len,
                output_script_len=0,
                model_max_tokens=max_tokens,
                model_name=self.model_name
            )
            return 0, ""

    def run(self) -> dict:
        start_time = time.time()
        if not self._init_model():
            failed_results = [0] * self.num_cases
            print(f"Test Results: {' '.join(str(r) for r in failed_results)} | Overall: 0")
            return {
                'model_name': self.model_name, 'success': False,
                'time_taken': time.time() - start_time, 'output': '',
                'case_results': failed_results
            }

        case_results = []
        model_results = []

        try:
            for i, prompt in enumerate(self.test_prompts, 1):
                case_success, predicted_text = self._process_case(i, prompt)
                case_results.append(case_success)
                model_results.append(predicted_text)

            overall_success = 1 if all(r == 1 for r in case_results) else 0
            total_elapsed = time.time() - start_time
            print(f"\n✓ {self.model_name}")
            print(f"  Test Results: {' '.join(str(r) for r in case_results)} | Overall: {overall_success}")
            print(f"  Total time: {total_elapsed:.2f}s")

            return {
                'model_name': self.model_name, 'success': overall_success == 1,
                'time_taken': total_elapsed, 'output': model_results,
                'case_results': case_results
            }
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ {self.model_name}: ERROR")
            print(f"  Error: {str(e)}")
            self.manager._log_error(self.model_name, str(e))
            failed_results = [0] * self.num_cases
            print(f"Test Results: {' '.join(str(r) for r in failed_results)} | Overall: 0")
            return {
                'model_name': self.model_name, 'success': False,
                'time_taken': elapsed, 'output': '',
                'case_results': failed_results
            }


def TextToText_main():
    root_folder = r"D:\AIs\Any-to-Any"
    manager = TextToTextModelFactory(root_folder)
    print("📦 Available models:")
    models = manager.list_available_models()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model.name} - {model.size_human}")
    if not models:
        print("❌ No models found. Check the path.")
        return

    for test_case_folder in [r"TestCases/Oracle/Basic",
                             # r"TestCases/Oracle/customer_orders",
                             r"TestCases/Oracle/human_resources"]:
        test_cases_loader = TestCasesLoaded(test_case_folder)
        apply_models_to_test_cases(manager, models, test_cases_loader)


def apply_models_to_test_cases(manager: TextToTextModelFactory, models: list[TextToTextModelInfo],
                               test_cases_loader: TestCasesLoaded):
    print(f"\n🚀 Starting text generation testing...")
    results = []
    num_cases = len(test_cases_loader.get_test_prompts())

    for model_info in models:
        model_name = model_info.name
        if model_name not in ALLOWED_MODELS:
            continue
        if manager.is_model_faulty(model_name):
            print(f"⊘ {model_name}: SKIPPED (error limit exceeded)")
            continue

        converter = CodeConverter(manager, model_info, test_cases_loader)
        result = converter.run()
        results.append(result)

    # Generate combined output files after all models have been processed
    test_cases_loader.save_combined_output_files()

    print(f"\n📊 Testing Summary")
    header = f"{'Model':<45}"
    for i in range(1, num_cases + 1):
        header += f" Case{i:<5}"
    header += f" {'Overall':<8} {'Time (s)':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        cases = r.get('case_results', [0] * num_cases)
        cases = (cases + [0] * num_cases)[:num_cases]
        overall = 1 if r['success'] else 0
        line = f"{r['model_name']:<45}"
        for c in cases:
            line += f" {c:<5}"
        line += f" {overall:<8} {r['time_taken']:<10.2f}"
        print(line)

    stats = {'test_type': 'text_to_sql_prompt', 'results': results}
    manager.save_statistics(stats)
    print(f"\n💾 Statistics saved to {manager.stats_path}")