from TextToText.CodeConverter import CodeConverter, KnowledgeMiner
from TextToText.TestCasesLoader import TestCasesLoaded
from TextToText.HtmlCasesLoaded import HtmlCasesLoaded
from TextToText.TextToTextModelFactory import TextToTextModelFactory
from TextToText.TextToTextModelInfo import TextToTextModelInfo

ALLOWED_MODELS = [
    # "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    # "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    # "Qwen/Qwen2.5-Coder-3B-Instruct",
    # "Qwen/Qwen2.5-Coder-7B-Instruct",
    # "Qwen/Qwen2.5-Coder-14B-Instruct",
    # "PipableAI/pip-sql-1.3b",
    # "prem-research/prem-1B-SQL",
    # "PipableAI/pip-sql-1.3b",
    "Bhuvneesh/gemma-4-E4B-it-Q8_0-GGUF",
    #"locailabs/Jupiter-G-8B",
    #"OneThink/OneThinker-8B",
]


def TextToText_main():
    root_folder = r"D:\AIs\Any-to-Any"
    manager = TextToTextModelFactory(root_folder)
    print("📦 Available models:")
    models = manager.list_available_models()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model.name} - {model.size_human}")
        #  ({model.path})
    for i, model in enumerate(models, 1):
        if model.size_bytes > 30 * 1024 * 1024 * 1024:
            dest_path = str(model.path).replace("D:", "E:")
            #print(f'Move-Item -Path "{model.path}" -Destination "{dest_path}" -Force')
            print(f'robocopy "{model.path}" "{dest_path}" /E /MOVE /MT:16 /R:3 /W:5')
    if not models:
        print("❌ No models found. Check the path.")
        return

    for test_case_folder in [
        #r"TestCases/HuggingFaceHtmls/G1"
        r"D:\AIs\Info"
    ]:
        test_cases_loader = HtmlCasesLoaded(test_case_folder)
        apply_models_to_htmls(manager, models, test_cases_loader)

    if False:
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

def apply_models_to_htmls(manager: TextToTextModelFactory,
                          models: list[TextToTextModelInfo],
                          test_cases_loader: TestCasesLoaded):
    print(f"\n🚀 Starting text generation ...")
    results = []
    num_cases = len(test_cases_loader.get_test_prompts())

    for model_info in models:
        model_name = model_info.name
        if model_name not in ALLOWED_MODELS:
            continue
        if manager.is_model_faulty(model_name):
            print(f"⊘ {model_name}: SKIPPED (error limit exceeded)")
            continue

        miner= KnowledgeMiner(manager, model_info, test_cases_loader)
        result = miner.run()
        results.append(result)

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

def collect_models_htmls_info(
        manager: TextToTextModelFactory,
        models: list[TextToTextModelInfo],
        test_cases_loader: TestCasesLoaded):
    print(f"\n🚀 Starting text generation ...")
    results = []
    num_cases = len(test_cases_loader.get_test_prompts())

    for model_info in models:
        model_name = model_info.name
        if model_name not in ALLOWED_MODELS:
            continue
        if manager.is_model_faulty(model_name):
            print(f"⊘ {model_name}: SKIPPED (error limit exceeded)")
            continue

        miner= KnowledgeMiner(manager, model_info, test_cases_loader)
        result = miner.run()
        results.append(result)

