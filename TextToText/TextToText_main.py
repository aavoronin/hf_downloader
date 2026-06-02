import time
from TextToText.TestCasesLoader import TestCasesLoaded
from TextToText.TextToTextModelFactory import TextToTextModelFactory, TextToTextModelInfo
from TextToText.OracleConverterHelper import OracleConverterHelper

ALLOWED_MODELS = [
    # "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    #"Qwen/Qwen2.5-Coder-3B-Instruct",
    #"Qwen/Qwen2.5-Coder-7B-Instruct",
    #"Qwen/Qwen2.5-Coder-14B-Instruct",
    # "PipableAI/pip-sql-1.3b",
    # "prem-research/prem-1B-SQL",
    # "PipableAI/pip-sql-1.3b",
]

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

    # test_prompts = OracleConverterHelper.get_test_prompts()
    for test_case_folder in [r"TestCases/Oracle/Basic",
    #r"TestCases/Oracle/customer_orders",
    r"TestCases/Oracle/human_resources"]:
        test_cases_loader = TestCasesLoaded(test_case_folder)
        test_prompts = test_cases_loader.get_test_prompts()
        apply_models_to_test_cases(manager, models, test_prompts, test_cases_loader)

def apply_models_to_test_cases(manager: TextToTextModelFactory, models: list[TextToTextModelInfo], test_prompts: list,
                               testCasesLoader: TestCasesLoaded):
    print(f"\n🚀 Starting text generation testing...")
    results = []
    num_cases = len(test_prompts)
    for model_info in models:
        model_name = model_info.name
        if model_name not in ALLOWED_MODELS:
            continue
        if manager.is_model_faulty(model_name):
            print(f"⊘ {model_name}: SKIPPED (error limit exceeded)")
            continue
        start_time = time.time()
        try:
            print(f"\n{'=' * 60}")
            print(f" Testing: {model_name}")
            print(f"{'=' * 60}")
            model = manager.create(model_name)
            if model is None:
                print(f"✗ {model_name}: FAILED to initialize")
                manager._log_error(model_name, "Initialization returned None")
                # Initialize case_results with zeros for all test cases
                failed_results = [0] * num_cases
                print(f"Test Results: {' '.join(str(r) for r in failed_results)} | Overall: 0")
                results.append({
                    'model_name': model_name, 'success': False,
                    'time_taken': time.time() - start_time, 'output': '',
                    'case_results': failed_results
                })
                continue
            case_results = []
            model_results = []
            for i, prompt in enumerate(test_prompts, 1):
                case_start = time.time()
                print(f"\n--- Test Case {i} ---")
                try:
                    predicted_text = model.process(prompt)
                    elapsed_case = time.time() - case_start
                    # Check if result is non-empty to count as success
                    case_success = 1 if predicted_text and len(predicted_text.strip()) > 0 else 0
                    case_results.append(case_success)
                    # Print output for each case individually, limited to 2000 symbols
                    output_preview = predicted_text[:2000] if predicted_text else ""
                    print(f"Result Case {i}:\n{output_preview}")
                    print(f"Time: {elapsed_case:.2f}s | Status: {case_success}")
                    model_results.append(predicted_text)
                    # Save result files to timestamped folder
                    prompt_len = len(prompt)
                    # Assuming prompt is [Base Prompt] + [SQL Input]
                    # Approximate input script length
                    input_script_len = prompt_len - len(
                        testCasesLoader.prompt_content) if testCasesLoader.prompt_content else prompt_len
                    # Try to get model limits from config if available
                    max_tokens = "Unknown"
                    if hasattr(model, '_custom_config') and model._custom_config:
                        max_tokens = str(model._custom_config.get('max_input_tokens', '?'))
                    testCasesLoader.save_test_case_result(
                        case_index=i - 1,
                        success=bool(case_success),
                        output_text=predicted_text,
                        time_taken=elapsed_case,
                        prompt_text=prompt,
                        input_script_len=input_script_len,
                        output_script_len=len(predicted_text) if predicted_text else 0,
                        model_max_tokens=max_tokens,
                        model_name=model_name
                    )
                except Exception as e:
                    print(f"Failed Case {i}: {str(e)}")
                    case_results.append(0)
                    model_results.append("")
                    # Save result files for failure
                    max_tokens = "Unknown"
                    if hasattr(model, '_custom_config') and model._custom_config:
                        max_tokens = str(model._custom_config.get('max_input_tokens', '?'))
                    testCasesLoader.save_test_case_result(
                        case_index=i - 1,
                        success=False,
                        output_text="",
                        time_taken=time.time() - case_start,
                        error_msg=str(e),
                        prompt_text=prompt,
                        input_script_len=len(prompt) - len(
                            testCasesLoader.prompt_content) if testCasesLoader.prompt_content else len(prompt),
                        output_script_len=0,
                        model_max_tokens=max_tokens,
                        model_name=model_name
                    )
            # Overall success only if ALL test cases succeeded
            overall_success = 1 if all(r == 1 for r in case_results) else 0
            total_elapsed = time.time() - start_time
            print(f"\n✓ {model_name}")
            print(f"  Test Results: {' '.join(str(r) for r in case_results)} | Overall: {overall_success}")
            print(f"  Total time: {total_elapsed:.2f}s")
            results.append({
                'model_name': model_name, 'success': overall_success == 1,
                'time_taken': total_elapsed, 'output': model_results,
                'case_results': case_results
            })
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ {model_name}: ERROR")
            print(f"  Error: {str(e)}")
            manager._log_error(model_name, str(e))
            # All cases failed due to exception
            failed_results = [0] * num_cases
            print(f"Test Results: {' '.join(str(r) for r in failed_results)} | Overall: 0")
            results.append({
                'model_name': model_name, 'success': False,
                'time_taken': elapsed, 'output': '',
                'case_results': failed_results
            })
    print(f"\n📊 Testing Summary")
    # Dynamic header generation based on number of test cases
    header = f"{'Model':<45}"
    for i in range(1, num_cases + 1):
        header += f" Case{i:<5}"
    header += f" {'Overall':<8} {'Time (s)':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        cases = r.get('case_results', [0] * num_cases)
        # Ensure cases length exactly matches num_cases
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