import re
import json
from pathlib import Path
from datetime import datetime
from sys import exception
from typing import List
from bs4 import BeautifulSoup, Comment, NavigableString

from TextToText.ModelFullInfo import ModelFullInfo


class TestCasesLoaded:
    BREAK_MARKER = '-- BREAK'

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.prompt_content = self._load_prompt()
        self.test_cases_data = self._load_test_cases()
        self.test_prompts = [tc["prompt"] for tc in self.test_cases_data]
        self.filenames = [tc["name"] for tc in self.test_cases_data]

        out_folder = Path("out") / self.folder_path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = out_folder / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_data = {}

    def _load_prompt(self) -> str:
        prompt_path = self.folder_path / "PROMPT.txt"
        if not prompt_path.exists():
            return ""
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_test_cases(self) -> list:
        if not self.prompt_content:
            return []
        sql_files = sorted(self.folder_path.glob("*.sql"))
        if not sql_files:
            return []
        cases = []
        for sql_file in sql_files:
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            if TestCasesLoaded.BREAK_MARKER in sql_content:
                parts = sql_content.split(TestCasesLoaded.BREAK_MARKER)
                for i, part in enumerate(parts, 1):
                    part_name = f"{sql_file.stem}_part{i:03d}"
                    cases.append({
                        "name": part_name,
                        "prompt": f"{self.prompt_content}\n====SCRIPT START====\n{part}\n====SCRIPT END====",
                        "sql": part
                    })
            else:
                cases.append({
                    "name": sql_file.stem,
                    "prompt": f"{self.prompt_content}\n====SCRIPT START====\n{sql_content}\n====SCRIPT END====",
                    "sql": sql_content
                })
        return cases

    def get_test_prompts(self) -> list:
        return self.test_prompts

    def save_test_case_result(self, case_index: int, success: bool, output_text: str,
                              time_taken: float, error_msg: str = "",
                              prompt_text: str = "", input_script_len: int = 0,
                              output_script_len: int = 0, model_max_tokens: str = "",
                              model_name: str = "", output_file_extension: str = "sql"):
        if 0 <= case_index < len(self.filenames):
            basename = self.filenames[case_index]
        else:
            basename = f"case_{case_index}"

        sql_file = self.output_dir / f"{basename}.{output_file_extension}"
        with open(sql_file, 'w', encoding='utf-8') as f:
            f.write(output_text if output_text else "")

        prompt_file = self.output_dir / f"{basename}_PROMPT.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_text)

        log_file = self.output_dir / f"{basename}.log"
        log_content = (
            f"Test Case: {basename}\n"
            f"Model: {model_name}\n"
            f"Status: {'Success' if success else 'Failure'}\n"
            f"Error: {error_msg if error_msg else 'None'}\n"
            f"Time Taken: {time_taken:.4f}s\n"
            f"Prompt Length: {len(prompt_text)}\n"
            f"Input Tokens (Approx): {len(prompt_text)}\n"
            f"Model Max Input/Output Tokens: {model_max_tokens}\n"
            f"Length of Input Script: {input_script_len}\n"
            f"Length of Output Script: {output_script_len}\n"
        )
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)

        original_sql = ""
        for tc in self.test_cases_data:
            if tc["name"] == basename:
                original_sql = tc["sql"]
                break
        escaped_original_sql = original_sql.replace("/*", "/ *").replace("*/", "* /")

        self.results_data[basename] = {
            "log_content": log_content,
            "original_sql": escaped_original_sql,
            "output_sql": output_text if output_text else "",
            "success": success,
            "time_taken": time_taken,
            "prompt_length": len(prompt_text),
            "input_tokens": len(prompt_text),
            "input_script_len": input_script_len,
            "output_script_len": output_script_len
        }

    def save_combined_output_files(self):
        if not self.results_data:
            return
        sorted_names = sorted(self.results_data.keys())
        num_test_cases = len(self.results_data)
        num_success = sum(1 for res in self.results_data.values() if res["success"])
        num_errors = num_test_cases - num_success
        total_time = sum(res["time_taken"] for res in self.results_data.values())
        total_prompt_len = sum(res["prompt_length"] for res in self.results_data.values())
        total_input_tokens = sum(res["input_tokens"] for res in self.results_data.values())
        total_input_script_len = sum(res["input_script_len"] for res in self.results_data.values())
        total_output_script_len = sum(res["output_script_len"] for res in self.results_data.values())

        combined_sql_path = self.output_dir / "all_test_cases.sql"
        with open(combined_sql_path, 'w', encoding='utf-8') as f:
            for name in sorted_names:
                res = self.results_data[name]
                f.write(f"-- {name}\n")
                f.write(f"{res['output_sql']}\n")

        combined_ext_sql_path = self.output_dir / "all_test_cases_ext.sql"
        with open(combined_ext_sql_path, 'w', encoding='utf-8') as f:
            f.write("/*\n")
            f.write(f"Number of Test Cases: {num_test_cases}\n")
            f.write(f"Number of Success: {num_success}\n")
            f.write(f"Number of Errors: {num_errors}\n")
            f.write(f"Total Time Taken: {total_time:.4f}s\n")
            f.write(f"Total Prompt Length: {total_prompt_len}\n")
            f.write(f"Total Input Tokens: {total_input_tokens}\n")
            f.write(f"Total Length of Input Script: {total_input_script_len}\n")
            f.write(f"Total Length of Output Script: {total_output_script_len}\n")
            f.write("========\n")
            f.write("*/\n")
            for name in sorted_names:
                res = self.results_data[name]
                f.write("/*\n")
                f.write(res["log_content"])
                f.write("\n")
                f.write(res["original_sql"])
                f.write("\n")
                f.write("*/\n")
                f.write("\n")
                f.write(f"{res['output_sql']}\n")


class HtmlCasesLoaded(TestCasesLoaded):
    BREAK_MARKER = '-- BREAK'
    USE_MARKUP_STRIPPING = True
    TOP_TAGS_COUNT = 300

    def __init__(self, folder_path: str, output_folder: str = r"D:\AIs\Info"):
        self.folder_path = Path(folder_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.text_to_text_models: List[ModelFullInfo] = []
        self.text_to_image_diffusion_models: List[ModelFullInfo] = []
        self.image_to_text_ocr_models: List[ModelFullInfo] = []
        self.text_image_to_text_nonocr_models: List[ModelFullInfo] = []

        self.test_cases_data = self._collect_existing_results()
        self.prompt_content = self._load_prompt()
        self.test_cases_data = self._load_html_cases()
        self.test_prompts = [tc["prompt"] for tc in self.test_cases_data]
        self.filenames = [tc["name"] for tc in self.test_cases_data]

        out_folder = Path("out") / self.folder_path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = out_folder / timestamp
        self.results_data = {}

    @staticmethod
    def _parse_size_to_bytes(size_str: str) -> int:
        if not size_str:
            return 0
        size_str = size_str.strip().upper().replace(',', '')
        match = re.match(r'^([0-9.]+)\s*(B|KB|MB|GB|TB|PB|KIB|MIB|GIB|TIB|PIB)$', size_str)
        if not match:
            return 0
        value = float(match.group(1))
        unit = match.group(2).replace('I', '')
        multipliers = {
            'B': 1, 'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3, 'TB': 1024 ** 4, 'PB': 1024 ** 5
        }
        return int(value * multipliers.get(unit, 1))

    def _get_model_files_info(self, folder_path: Path) -> tuple[str, str]:
        json_path = folder_path / "model_files_page.json"
        html_path = folder_path / "model_files_page.html"
        size_str = ""
        size_bytes_str = ""
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                size_str = data.get("Size", "")
                size_bytes_str = data.get("SizeB", "")
                return size_str, size_bytes_str
            except Exception:
                pass
        if html_path.exists():
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    files_html_content = f.read()
                soup_files = BeautifulSoup(files_html_content, 'html.parser')
                target_div = soup_files.find('div', class_=lambda
                    c: c and 'py-[3px]' in c and 'font-mono' in c and 'text-gray-500' in c)
                if target_div:
                    size_str = target_div.get_text(strip=True)
                    size_bytes = self._parse_size_to_bytes(size_str)
                    size_bytes_str = str(size_bytes)
                    try:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump({"Size": size_str, "SizeB": size_bytes_str}, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
            except Exception:
                pass
        return size_str, size_bytes_str

    def _get_model_exact_tags(self, html_file: Path, model_id: str) -> list[str]:
        json_path = html_file.parent / "model_tags.json"
        cutoff_date = datetime(2026, 7, 13, 20, 16, 0)
        if json_path.exists():
            try:
                mod_time = datetime.fromtimestamp(json_path.stat().st_mtime)
                if mod_time > cutoff_date:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        tags = json.load(f)
                    if isinstance(tags, list):
                        return tags
            except Exception:
                pass
        tags = []
        if html_file.exists():
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                for a_tag in soup.find_all('a', href=lambda h: h and h.startswith('/models?')):
                    span = a_tag.find('span')
                    if span:
                        tag_text = span.get_text(strip=True)
                        if tag_text:
                            tags.append(tag_text)
                if model_id:
                    parts = re.split(r'[-/\\_]', model_id)
                    for part in parts:
                        part = part.strip()
                        if not part: continue
                        if len(part) <= 1: continue
                        if not re.search(r'[a-zA-Z]', part): continue
                        part_lower = part.lower()
                        if re.match(r'^\d+(\.\d+)?$', part): continue
                        if re.match(r'^v\d+(\.\d+)?$', part_lower): continue
                        if re.match(r'^\d+(\.\d+)?[bmg]$', part_lower): continue
                        if part not in tags: tags.append(part)
                tags = list(set([t.lower() for t in tags]))
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(tags, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            except Exception:
                pass
        return tags

    def _collect_models_of_interest(self, item: dict, sorted_tags: list):
        model_info_data = item.get("model_info", {})
        model_id = model_info_data.get("model_id", "") or ""
        if not model_id:
            return

        file_path = item.get("file_path", "Unknown Path")
        has_code = item.get("has_code", False)
        size_str = item.get("size_str", "")
        size_bytes_str = item.get("size_bytes_str", "")

        size_bytes_for_range = int(size_bytes_str) if size_bytes_str and size_bytes_str.isdigit() else 0
        size_range = self.get_size_range(size_bytes_for_range)

        # Limit model sizes to 20 Gb
        if size_bytes_for_range > 20 * 1024 ** 3:
            return

        size_bytes_int = int(size_bytes_str) if size_bytes_str and size_bytes_str.isdigit() else None
        exact_tags = item.get("exact_tags", [])
        exact_tags_lower = [t.lower() for t in exact_tags]
        downloads = model_info_data.get("downloads", 0) or 0
        likes = model_info_data.get("likes", 0) or 0

        def escape_csv(val):
            return str(val).replace('"', '""')

        model_url = f"https://huggingface.co/{escape_csv(model_id)}" if model_id else ""

        data = item.get("json", {})
        model_size = str(data.get("model_size")) if data.get("model_size") is not None else ""
        input_mods = data.get("input_modalities") or []
        output_mods = data.get("output_modalities") or []
        input_modalities = ",".join(str(m) for m in input_mods) if isinstance(input_mods, list) else ""
        output_modalities = ",".join(str(m) for m in output_mods) if isinstance(output_mods, list) else ""

        text_i = "1" if "Text" in input_mods else ""
        image_i = "1" if "Image" in input_mods else ""
        audio_i = "1" if "Audio" in input_mods else ""
        video_i = "1" if "Video" in input_mods else ""

        text_o = "1" if "Text" in output_mods else ""
        image_o = "1" if "Image" in output_mods else ""
        audio_o = "1" if "Audio" in output_mods else ""
        video_o = "1" if "Video" in output_mods else ""
        three_d_o = "1" if "3D" in output_mods else ""

        def get_int_val(val):
            if val is None:
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        input_tokens = get_int_val(data.get("input_tokens"))
        output_tokens = get_int_val(data.get("output_tokens"))
        code = data.get("code", "")

        model_full_info = ModelFullInfo(
            model_id=model_id,
            file_path=file_path,
            model_url=model_url,
            has_code=has_code,
            Size=size_str,
            SizeRange=size_range,
            input_modalities=input_modalities,
            Text_I=text_i,
            Image_I=image_i,
            Audio_I=audio_i,
            Video_I=video_i,
            output_modalities=output_modalities,
            Text_O=text_o,
            Image_O=image_o,
            Audio_O=audio_o,
            Video_O=video_o,
            three_d_O=three_d_o,
            model_size=model_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            downloads=downloads,
            likes=likes,
            SizeB=size_bytes_int,
            code=code,
            sorted_tags=sorted_tags
        )

        input_mods_set = set(input_mods)
        output_mods_set = set(output_mods)

        if input_mods_set == {"Text"} and output_mods_set == {"Text"}:
            self.text_to_text_models.append(model_full_info)

        if input_mods_set == {"Text"} and output_mods_set == {"Image"} and "diffusers" in exact_tags_lower:
            self.text_to_image_diffusion_models.append(model_full_info)

        if input_mods_set == {"Image"} and output_mods_set == {"Text"} and "ocr" in exact_tags_lower:
            self.image_to_text_ocr_models.append(model_full_info)

        if "Image" in input_mods_set and input_mods_set <= {"Text", "Image"} and output_mods_set == {
            "Text"} and "ocr" not in exact_tags_lower:
            self.text_image_to_text_nonocr_models.append(model_full_info)

    def print_collection(self, name, collection):
        print(f"\ncollection {name} ({len(collection)} models):")
        for m in collection:
            size_gb = m.SizeB / (1024 ** 3) if m.SizeB is not None else 0.0
            print(f"{m.model_id} -- {size_gb:.2f} Gb")

        combined_information_file_name = self.output_folder / f"{name}.txt"


    def _collect_existing_results(self):
        html_files = self.collect_case_files()
        if not html_files:
            return []

        existing_data = []
        tag_counts = {}

        for i, html_file in enumerate(html_files):
            if i % 20 == 0:
                print(f"{i:>6} {html_file}")

            model_page_path = html_file.parent / "model_page.json"
            if not model_page_path.exists():
                continue

            model_info_path = html_file.parent / "model_info.json"
            model_id = ""
            downloads = 0
            likes = 0
            if model_info_path.exists():
                try:
                    with open(model_info_path, 'r', encoding='utf-8') as f:
                        info_data = json.load(f)
                    model_id = info_data.get("Model ID", "") or ""
                    downloads = info_data.get("Downloads", 0) or 0
                    likes = info_data.get("Likes", 0) or 0
                except Exception:
                    pass

            size_str, size_bytes_str = self._get_model_files_info(html_file.parent)
            exact_tags = self._get_model_exact_tags(html_file, model_id)

            for tag in exact_tags:
                tag_lower = tag.lower()
                tag_counts[tag_lower] = tag_counts.get(tag_lower, 0) + 1

            try:
                with open(model_page_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                has_code = False
                code_field = data.get("code", "")
                if isinstance(code_field, str) and len(code_field) > 50:
                    has_code = True

                existing_data.append({
                    "file_path": str(model_page_path),
                    "json": data,
                    "model_info": {"model_id": model_id, "downloads": downloads, "likes": likes},
                    "has_code": has_code,
                    "size_str": size_str,
                    "size_bytes_str": size_bytes_str,
                    "exact_tags": exact_tags
                })
            except json.JSONDecodeError as e:
                print(f"Warning: {model_page_path} is not valid JSON. Error: {e}")
                existing_data.append({
                    "file_path": str(model_page_path),
                    "error": f"Invalid JSON: {str(e)}",
                    "model_info": {"model_id": model_id, "downloads": downloads, "likes": likes},
                    "has_code": False,
                    "size_str": size_str,
                    "size_bytes_str": size_bytes_str,
                    "exact_tags": exact_tags
                })
            except Exception as e:
                existing_data.append({
                    "file_path": str(model_page_path),
                    "error": f"Unexpected Error: {str(e)}",
                    "model_info": {"model_id": model_id, "downloads": downloads, "likes": likes},
                    "has_code": False,
                    "size_str": size_str,
                    "size_bytes_str": size_bytes_str,
                    "exact_tags": exact_tags
                })

        sorted_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)
        all_top_tags = [tag for tag, count in sorted_tags[:self.TOP_TAGS_COUNT]]
        pattern = re.compile(r'^([a-z0-9]+-)+to(-[a-z0-9]+)+$')
        group_to = [tag for tag in all_top_tags if pattern.match(tag)]
        group_other = [tag for tag in all_top_tags if not pattern.match(tag)]
        group_to.sort()
        group_other.sort()
        top_tags = group_to + group_other
        top_tags_set = set(top_tags)

        print(f"\nTotal records loaded: {len(existing_data)} remaining: {len(html_files) - len(existing_data)}")
        print("=" * 120)

        csv_file_path = self.output_folder / "models_summary.csv"
        with open(csv_file_path, 'w', encoding='utf-8') as csv_file:
            def escape_csv(val):
                return str(val).replace('"', '""')

            def get_int_token(val):
                if val is None: return ""
                try:
                    return str(int(val))
                except (ValueError, TypeError):
                    return ""

            base_header = 'row_num,file_path,model_url,model_id,has_code,Size,SizeRange,input_modalities,Text_I,Image_I,Audio_I,Video_I,output_modalities,Text_O,Image_O,Audio_O,Video_O,3D_O,model_size,input_tokens,output_tokens,downloads,likes,SizeB'
            header = base_header + ',' + ','.join(top_tags) + ',RemainingTags'
            print(header)
            csv_file.write(header + '\n')

            for row_num, item in enumerate(existing_data, start=1):
                file_path = item.get("file_path", "Unknown Path")
                model_info_data = item.get("model_info", {})
                has_code = item.get("has_code", False)
                size_str = item.get("size_str", "")
                size_bytes_str = item.get("size_bytes_str", "")
                size_bytes = int(size_bytes_str) if size_bytes_str.isdigit() else 0
                size_range = self.get_size_range(size_bytes)
                exact_tags = item.get("exact_tags", [])
                exact_tags_lower = [t.lower() for t in exact_tags]
                model_id = model_info_data.get("model_id", "") or ""
                downloads = model_info_data.get("downloads", 0) or 0
                likes = model_info_data.get("likes", 0) or 0
                model_url = f"https://huggingface.co/{escape_csv(model_id)}" if model_id else ""

                if "error" in item:
                    err_msg = escape_csv(item['error'])
                    empty_cols = ",".join(['""'] * 12)
                    base_row = f'{row_num},"{escape_csv(file_path)}","{model_url}","{escape_csv(model_id)}","{has_code}","{escape_csv(size_str)}","{size_range}","ERROR","{err_msg}",{empty_cols},{downloads},{likes},{size_bytes_str}'
                    row = base_row + "," + ",".join([""] * len(top_tags)) + ',\"\"'
                    csv_file.write(row + '\n')
                    continue

                data = item.get("json", {})
                model_size = str(data["model_size"]) if data.get("model_size") is not None else ""
                input_mods = data.get("input_modalities") or []
                output_mods = data.get("output_modalities") or []
                input_modalities = ",".join(str(m) for m in input_mods) if isinstance(input_mods, list) else ""
                output_modalities = ",".join(str(m) for m in output_mods) if isinstance(output_mods, list) else ""
                text_i = "1" if "Text" in input_mods else ""
                image_i = "1" if "Image" in input_mods else ""
                audio_i = "1" if "Audio" in input_mods else ""
                video_i = "1" if "Video" in input_mods else ""
                text_o = "1" if "Text" in output_mods else ""
                image_o = "1" if "Image" in output_mods else ""
                audio_o = "1" if "Audio" in output_mods else ""
                video_o = "1" if "Video" in output_mods else ""
                three_d_o = "1" if "3D" in output_mods else ""
                input_tokens = get_int_token(data.get("input_tokens"))
                output_tokens = get_int_token(data.get("output_tokens"))

                base_row = f'{row_num},"{escape_csv(file_path)}","{model_url}","{escape_csv(model_id)}","{has_code}","{escape_csv(size_str)}","{size_range}","{escape_csv(input_modalities)}","{text_i}","{image_i}","{audio_i}","{video_i}","{escape_csv(output_modalities)}","{text_o}","{image_o}","{audio_o}","{video_o}","{three_d_o}","{escape_csv(model_size)}","{input_tokens}","{output_tokens}",{downloads},{likes},{size_bytes_str}'
                tag_cols = ["1" if t in exact_tags_lower else "" for t in top_tags]
                row = base_row + "," + ",".join(tag_cols)
                remaining_tags = [t for t in exact_tags if t.lower() not in top_tags_set]
                row += "," + escape_csv("|".join(remaining_tags))
                csv_file.write(row + '\n')

                self._collect_models_of_interest(item, sorted_tags)

        print("=" * 120)
        print(f"Processing complete. CSV saved to {csv_file_path}")

        # ==========================================
        # PRINT COLLECTIONS OF INTEREST
        # ==========================================
        print("\n" + "=" * 80)
        print("COLLECTIONS OF INTEREST")
        print("=" * 80)

        self.print_collection("text_to_text_models", self.text_to_text_models)
        self.print_collection("text_to_image_diffusion_models", self.text_to_image_diffusion_models)
        self.print_collection("image_to_text_ocr_models", self.image_to_text_ocr_models)
        self.print_collection("text_image_to_text_nonocr_models", self.text_image_to_text_nonocr_models)

        if tag_counts:
            print("\n🏷️ TAG STATISTICS (Sorted by frequency)")
            print("-" * 40)
            sorted_tags_print = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)
            for i, (tag, count) in enumerate(sorted_tags_print):
                print(f"{i:>6} {tag}: {count}")

    def collect_case_files(self) -> list[Path]:
        return sorted(self.folder_path.rglob("model_page.html"))

    def get_size_range(self, size_bytes: int) -> str:
        if size_bytes == 0: return ""
        SIZE_RANGES = [
            (0.5 * 1024 ** 3, "0.5Gb"), (1 * 1024 ** 3, "1Gb"), (2 * 1024 ** 3, "2Gb"), (3 * 1024 ** 3, "3Gb"),
            (4 * 1024 ** 3, "4Gb"), (5 * 1024 ** 3, "5Gb"), (7.5 * 1024 ** 3, "7.5Gb"), (10 * 1024 ** 3, "10Gb"),
            (15 * 1024 ** 3, "15Gb"), (20 * 1024 ** 3, "20Gb"), (30 * 1024 ** 3, "30Gb"), (50 * 1024 ** 3, "50Gb"),
            (75 * 1024 ** 3, "75Gb"), (100 * 1024 ** 3, "100Gb"), (200 * 1024 ** 3, "200Gb"),
            (300 * 1024 ** 3, "300Gb"),
            (500 * 1024 ** 3, "500Gb"), (1024 ** 4, "1Tb"), (2 * 1024 ** 4, "2Tb"), (3 * 1024 ** 4, "3Tb"),
            (4 * 1024 ** 4, "4Tb"), (5 * 1024 ** 4, "5Tb"),
        ]
        for threshold, label in SIZE_RANGES:
            if size_bytes <= threshold: return label
        return "5Tb+"

    def _load_html_cases(self) -> list:
        if not self.prompt_content: return []
        html_files = self.collect_case_files()
        if not html_files: return []
        cases = []
        for i, html_file in enumerate(html_files):
            model_page_path = html_file.parent / "model_page.json"
            model_page_path_txt = html_file.parent / "model_page.txt"
            if model_page_path_txt.exists(): continue
            if model_page_path.exists():
                try:
                    target_date = datetime(2026, 6, 20, 14, 0, 0)
                    if datetime.fromtimestamp(Path(model_page_path).stat().st_mtime) >= target_date:
                        with open(model_page_path, 'r', encoding='utf-8') as f:
                            if "model_name" in json.load(f): continue
                except Exception:
                    pass
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            html_content = self.html_to_formatted_text(html_content) if self.USE_MARKUP_STRIPPING else self.clean_html(
                html_content, html_file)
            downloads, likes = 0, 0
            model_info_path = html_file.parent / "model_info.json"
            if model_info_path.exists():
                try:
                    with open(model_info_path, 'r', encoding='utf-8') as f:
                        model_info = json.load(f)
                    downloads, likes = model_info.get("Downloads", 0), model_info.get("Likes", 0)
                except exception:
                    pass
            cases.append(
                {"name": html_file.stem, "prompt": f"{self.prompt_content}\n{html_content}\n", "path": html_file,
                 "downloads": downloads, "likes": likes})
        cases.sort(key=lambda x: x["downloads"] + x["likes"] * 200, reverse=True)
        return cases

    def clean_html(self, html_content, html_file):
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag_name in ['head', 'script', 'svg', 'path', 'defs']:
            for tag in soup.find_all(tag_name): tag.decompose()
        for tag in soup.find_all(True):
            for attr in ['style', 'class', 'id', 'xmlns', 'rel']:
                if tag.has_attr(attr): del tag[attr]
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)): comment.extract()
        for tag in soup.find_all(href=True):
            if 'huggingface' not in tag['href'] and 'github' not in tag['href']: del tag['href']
        return str(soup)

    def html_to_formatted_text(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup.find_all(['img', 'script', 'style', 'noscript', 'svg']): tag.decompose()
        for tag in soup.find_all(style=re.compile(r'display\s*:\s*none', re.I)): tag.decompose()
        for tag in soup.find_all(hidden=True): tag.decompose()
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(separator=' ', strip=True).replace('|', '/') for td in tr.find_all(['th', 'td'])]
                if cells: rows.append("| " + " | ".join(cells) + " |")
            table.replace_with(NavigableString(f"\n[Table Start]\n" + "\n".join(rows) + "\n[Table End]\n"))
        for i in range(1, 7):
            for header in soup.find_all(f'h{i}'):
                header.replace_with(NavigableString(f"\n{'=' * i} {header.get_text(strip=True)} {'=' * i}\n"))
        for li in soup.find_all('li'): li.insert(0, NavigableString("* "))
        text = soup.get_text(separator='\n')
        return re.sub(r'\n{3,}', '\n', text).strip()