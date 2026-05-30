import os
import time
from TextToText.TextToTextModelFactory import TextToTextModelFactory

ALLOWED_MODELS = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    #"Qwen/Qwen2.5-Coder-14B-Instruct",
    "PipableAI/pip-sql-1.3b",
    "prem-research/prem-1B-SQL",
    "PipableAI/pip-sql-1.3b",
]

def TextToText_main():
    root_folder = r"D:\AIs\Any-to-Any"
    manager = TextToTextModelFactory(root_folder)
    print("📦 Available models:")
    models = manager.list_available_models()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model.name} - {model.size_human}")

    instruction = """
You are a Data Engineer who converts SQL from Oracle to Arenadata Greenplum version 6.30.
You are given SQL query or SQL script in oracle you should produce equivalent script in Arenadata Greenplum version 6.30.
Your output should contain the converted query or sql script only.
Comments comments should ber included if they have been in the original query.
No explanations or statements on what you have done or how you have done it should be included.
No recommendations or proposals should be included. 
Do not transfer COMMIT statements to Greenplum if hey are not needed.
Use ZLIB(5) compression in Greenplum. 
Unless specifically instructed use "distributed randomly" on create tables.  

Original prompt or any of it's parts should not be included.  
Use the following guidelines. 
Convert types as follows.  
CHAR -> char
VARCHAR2 -> text
NVARCHAR2 -> text
NCHAR -> char
CLOB -> text
NCLOB -> text
LONG -> text
NUMBER -> numeric
BINARY_FLOAT -> numeric
BINARY_DOUBLE -> numeric
FLOAT -> numeric
DECIMAL -> numeric
INTEGER -> numeric
DATE -> timestamp
TIMESTAMP -> timestamp
TIMESTAMP WITH TIME ZONE -> timestamptz
TIMESTAMP WITH LOCAL TIME ZONE -> timestamp
INTERVAL YEAR TO MONTH -> interval
INTERVAL DAY TO SECOND -> interval
RAW -> bytea
LONG RAW -> bytea
BLOB -> bytea
ROWID -> text
UROWID -> text
XMLTYPE -> xml
BFILE -> text
BOOLEAN -> boolean
ORACLE_TO_GREENPLUM_NAMING_CONVENTION:
CASE: Convert all identifiers to lowercase.
MAX_LENGTH: 63 characters. Truncate from right if exceeded. Append _1, _2, etc. if duplicates occur after truncation.
ALLOWED_CHARS: Only a-z, 0-9, _. Replace all other characters with _.
START_CHAR: Must begin with lowercase letter or _. Prepend tbl_ or col_ if starting character is invalid.
ORACLE_SPECIAL: Replace $ and # with _. Collapse consecutive underscores to single _.
RESERVED_WORDS: If identifier matches Greenplum/PostgreSQL reserved keyword, append _tbl for tables/views or _col for columns.
SCHEMA: Map Oracle schema/user name directly to Greenplum schema name, applying identical rules.
SCOPE: Apply uniformly to tables, views, columns, indexes, constraints, sequences, and functions.
QUOTING: Avoid double-quoting. Prefer unquoted lowercase identifiers. Quote only when reserved word conflict cannot be resolved by suffixing.
        """
    test_cases_sql = [
        "CREATE TABLE demo_table (\n    id_int   NUMBER(10),\n    val_float NUMBER(12,4),\n    txt_var  VARCHAR2(255)\n);\nCOMMIT;",
        "SELECT \n    42            AS int_val,\n    3.14159       AS float_val,\n    'Oracle Text' AS string_val\nFROM DUAL;",
        "INSERT INTO demo_table (id_int, val_float, txt_var)\nVALUES (101, 99.9500, 'Inserted row');\nCOMMIT;",
        "DELETE FROM demo_table;\nCOMMIT;"
    ]
    test_prompts = [instruction + case for case in test_cases_sql]

    if not models:
        print("❌ No models found. Check the path.")
        return

    print(f"\n🚀 Starting text generation testing...")
    results = []

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
                # All 4 cases failed due to init failure
                print(f"Test Results: 0 0 0 0 | Overall: 0")
                results.append({
                    'model_name': model_name, 'success': False,
                    'time_taken': time.time() - start_time, 'output': '',
                    'case_results': [0, 0, 0, 0]
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
                except Exception as e:
                    print(f"Failed Case {i}: {str(e)}")
                    case_results.append(0)
                    model_results.append("")

            # Overall success only if ALL 4 test cases succeeded
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
            print(f"Test Results: 0 0 0 0 | Overall: 0")
            results.append({
                'model_name': model_name, 'success': False,
                'time_taken': elapsed, 'output': '',
                'case_results': [0, 0, 0, 0]
            })

    print(f"\n📊 Testing Summary")
    print(f"{'Model':<45} {'Case1':<6} {'Case2':<6} {'Case3':<6} {'Case4':<6} {'Overall':<8} {'Time (s)':<10}")
    print("-" * 95)
    for r in results:
        cases = r.get('case_results', [0, 0, 0, 0])
        overall = 1 if r['success'] else 0
        print(f"{r['model_name']:<45} {cases[0]:<6} {cases[1]:<6} {cases[2]:<6} {cases[3]:<6} {overall:<8} {r['time_taken']:<10.2f}")

    stats = {'test_type': 'text_to_sql_prompt', 'results': results}
    manager.save_statistics(stats)
    print(f"\n💾 Statistics saved to {manager.stats_path}")