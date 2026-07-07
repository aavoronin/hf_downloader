import time

from TextToText.TestCasesLoader import TestCasesLoaded
from TextToText.TextToTextModelFactory import TextToTextModelFactory
from TextToText.TextToTextModelInfo import TextToTextModelInfo


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



class KnowledgeMiner:
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
            print(f"{i:>6} Time: {elapsed_case:.2f}s | Status: {case_success} | prompt len: {len(prompt)}")

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
                'case_results': case_results,
                'prompt': prompt
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
