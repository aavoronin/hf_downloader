import os
import time
from TextToText.TextToTextModelFactory import TextToTextModelFactory

def TextToText_main():
    root_folder = r"D:\AIs\text-to-sql"
    manager = TextToTextModelFactory(root_folder)
    print("📦 Доступные модели:")
    models = manager.list_available_models()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model.name} - {model.size_human}")

    test_prompt = (
        "Convert from Oracle from Greenplum 6.30\n"
        "=========TEST CASE========\n"
        "-- ==========================================\n"
        "-- 1) CREATE TABLE\n"
        "-- ==========================================\n"
        "CREATE TABLE demo_table (\n"
        "    id_int   NUMBER(10),\n"
        "    val_float NUMBER(12,4),\n"
        "    txt_var  VARCHAR2(255)\n"
        ");\n"
        "COMMIT;\n"
        "-- ==========================================\n"
        "-- 2) SELECT int, float, string FROM DUAL\n"
        "-- ==========================================\n"
        "SELECT \n"
        "    42            AS int_val,\n"
        "    3.14159       AS float_val,\n"
        "    'Oracle Text' AS string_val\n"
        "FROM DUAL;\n"
        "-- ==========================================\n"
        "-- 3) INSERT ONE ROW INTO TABLE (from #1)\n"
        "-- ==========================================\n"
        "INSERT INTO demo_table (id_int, val_float, txt_var)\n"
        "VALUES (101, 99.9500, 'Inserted row');\n"
        "COMMIT;\n"
        "-- ==========================================\n"
        "-- 4) DELETE FROM TABLE (from #1)\n"
        "-- ==========================================\n"
        "DELETE FROM demo_table;\n"
        "COMMIT;\n"
        "=========END TEST CASE========="
    )

    if not models:
        print("❌ Модели не найдены. Проверьте путь.")
        return

    print(f"\n🚀 Запуск тестирования генерации текста...")
    print(f"📄 Длина запроса: {len(test_prompt)} символов\n")

    results = []
    for model_info in models:
        model_name = model_info.name
        if manager.is_model_faulty(model_name):
            print(f"⊘ {model_name}: ПРОПУЩЕНО (превышен лимит ошибок)")
            continue

        start_time = time.time()
        try:
            print(f"\n{'=' * 60}")
            print(f" Тестирование: {model_name}")
            print(f"{'=' * 60}")
            model = manager.create(model_name)
            if model is None:
                print(f"✗ {model_name}: НЕ УДАЛОСЬ инициализировать")
                manager._log_error(model_name, "Инициализация вернула None")
                results.append({
                    'model_name': model_name, 'success': False,
                    'time_taken': time.time() - start_time, 'output': ''
                })
                continue

            predicted_text = model.process(test_prompt)
            elapsed = time.time() - start_time

            if not predicted_text or len(predicted_text.strip()) == 0:
                print(f"⚠ {model_name}: Пустой вывод")
                manager._log_error(model_name, "Пустой вывод")
                results.append({
                    'model_name': model_name, 'success': False,
                    'time_taken': elapsed, 'output': ''
                })
                continue

            print(f"\n📝 Вывод (первые 200 символов): {predicted_text[:200]}...")
            print(f"\n✓ {model_name}")
            print(f"  Время: {elapsed:.2f}с")
            results.append({
                'model_name': model_name, 'success': True,
                'time_taken': elapsed, 'output': predicted_text
            })
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ {model_name}: ОШИБКА")
            print(f"  Ошибка: {str(e)}")
            manager._log_error(model_name, str(e))
            results.append({
                'model_name': model_name, 'success': False,
                'time_taken': elapsed, 'output': ''
            })

    print(f"\n📊 Итоги тестирования")
    print(f"{'Модель':<45} {'Статус':<10} {'Время (с)':<10}")
    print("-" * 65)
    for r in results:
        status = "УСПЕХ" if r['success'] else "ОШИБКА"
        print(f"{r['model_name']:<45} {status:<10} {r['time_taken']:<10.2f}")

    stats = {'test_type': 'text_to_sql_prompt', 'results': results}
    manager.save_statistics(stats)
    print(f"\n💾 Статистика сохранена в {manager.stats_path}")
