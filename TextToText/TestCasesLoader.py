import re
import json
from pathlib import Path
from datetime import datetime
from sys import exception
from bs4 import BeautifulSoup, Comment, NavigableString


class TestCasesLoaded:
    BREAK_MARKER = '\n-- BREAK'

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.prompt_content = self._load_prompt()
        self.test_cases_data = self._load_test_cases()
        self.test_prompts = [tc["prompt"] for tc in self.test_cases_data]
        self.filenames = [tc["name"] for tc in self.test_cases_data]

        # Create output folder structure
        # out_folder = f"out/folder_path" -> e.g. out/TestCases/Oracle/Basic
        out_folder = Path("out") / self.folder_path

        # Create timestamp folder YYYYMMDDHHMMSS
        # YYYYMMDDHHMMSS is remembered when the group of test cases is started
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = out_folder / timestamp

        # Create the directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store results for combined output files
        self.results_data = {}

    def _load_prompt(self) -> str:
        """Load PROMPT.txt content. Returns empty string if not found."""
        prompt_path = self.folder_path / "PROMPT.txt"
        if not prompt_path.exists():
            return ""
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_test_cases(self) -> list:
        """Scan for .sql files and attach prompt to each."""
        if not self.prompt_content:
            return []

        sql_files = sorted(self.folder_path.glob("*.sql"))
        if not sql_files:
            return []

        cases = []
        for sql_file in sql_files:
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()

            # Check for the break marker and split into virtual parts if found
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
        """Return the list of combined prompt + SQL test cases."""
        return self.test_prompts

    def save_test_case_result(self, case_index: int, success: bool, output_text: str,
                              time_taken: float, error_msg: str = "",
                              prompt_text: str = "", input_script_len: int = 0,
                              output_script_len: int = 0, model_max_tokens: str = "",
                              model_name: str = "",
                              output_file_extension: str = "sql"):
        """
        Saves the 3 result files for a specific test case into the timestamped output directory.
        Also stores data for combined output files.
        """
        if 0 <= case_index < len(self.filenames):
            basename = self.filenames[case_index]
        else:
            basename = f"case_{case_index}"

        # 1. test_case.sql (Result) -> basename.sql
        sql_file = self.output_dir / f"{basename}.{output_file_extension}"
        with open(sql_file, 'w', encoding='utf-8') as f:
            f.write(output_text if output_text else "")

        # 2. test_case_PROMPT.txt (Full prompt) -> basename_PROMPT.txt
        prompt_file = self.output_dir / f"{basename}_PROMPT.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_text)

        # 3. test_case.log -> basename.log
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

        # Find original SQL for this case
        original_sql = ""
        for tc in self.test_cases_data:
            if tc["name"] == basename:
                original_sql = tc["sql"]
                break

        # Escape nested comments in original query
        escaped_original_sql = original_sql.replace("/*", "/ *").replace("*/", "* /")

        # Store for combined files generation
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
        """
        Generates all_test_cases.sql and all_test_cases_ext.sql
        containing the model outputs, sorted ASC by test case name.
        """
        if not self.results_data:
            return

        # Sort test cases ASC by name
        sorted_names = sorted(self.results_data.keys())

        # Calculate total statistics
        num_test_cases = len(self.results_data)
        num_success = sum(1 for res in self.results_data.values() if res["success"])
        num_errors = num_test_cases - num_success
        total_time = sum(res["time_taken"] for res in self.results_data.values())
        total_prompt_len = sum(res["prompt_length"] for res in self.results_data.values())
        total_input_tokens = sum(res["input_tokens"] for res in self.results_data.values())
        total_input_script_len = sum(res["input_script_len"] for res in self.results_data.values())
        total_output_script_len = sum(res["output_script_len"] for res in self.results_data.values())

        # Create all_test_cases.sql
        combined_sql_path = self.output_dir / "all_test_cases.sql"
        with open(combined_sql_path, 'w', encoding='utf-8') as f:
            for name in sorted_names:
                res = self.results_data[name]
                f.write(f"-- {name}\n")
                f.write(f"{res['output_sql']}\n")

        # Create all_test_cases_ext.sql
        combined_ext_sql_path = self.output_dir / "all_test_cases_ext.sql"
        with open(combined_ext_sql_path, 'w', encoding='utf-8') as f:
            # Add total statistics at the beginning
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
    BREAK_MARKER = '\n-- BREAK'
    USE_MARKUP_STRIPPING = True
    TOP_TAGS_COUNT = 300

    def __init__(self, folder_path: str, output_folder: str = r"D:\AIs\Info"):
        self.folder_path = Path(folder_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.test_cases_data = self._collect_existing_results()
        self.prompt_content = self._load_prompt()
        self.test_cases_data = self._load_html_cases()
        self.test_prompts = [tc["prompt"] for tc in self.test_cases_data]
        self.filenames = [tc["name"] for tc in self.test_cases_data]

        # Create output folder structure
        # out_folder = f"out/folder_path" -> e.g. out/TestCases/Oracle/Basic
        out_folder = Path("out") / self.folder_path

        # Create timestamp folder YYYYMMDDHHMMSS
        # YYYYMMDDHHMMSS is remembered when the group of test cases is started
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = out_folder / timestamp

        # Create the directory structure
        # self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store results for combined output files
        self.results_data = {}

    @staticmethod
    def _parse_size_to_bytes(size_str: str) -> int:
        """Convert human-readable size string (e.g., '68.8 GB') to bytes."""
        if not size_str:
            return 0
        size_str = size_str.strip().upper().replace(',', '')
        match = re.match(r'^([0-9.]+)\s*(B|KB|MB|GB|TB|PB|KIB|MIB|GIB|TIB|PIB)$', size_str)
        if not match:
            return 0

        value = float(match.group(1))
        unit = match.group(2).replace('I', '')  # Normalize GiB to GB, etc.

        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
            'TB': 1024 ** 4,
            'PB': 1024 ** 5
        }
        return int(value * multipliers.get(unit, 1))

    def _get_model_files_info(self, folder_path: Path) -> tuple[str, str]:
        """
        Load or parse model_files_page to extract Size and SizeB.
        Checks for model_files_page.json first. If not found, parses
        model_files_page.html and saves the result to JSON.
        Returns a tuple of (size_str, size_bytes_str).
        """
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

                # Find the specific div containing the size based on unique Tailwind classes
                target_div = soup_files.find(
                    'div',
                    class_=lambda c: c and 'py-[3px]' in c and 'font-mono' in c and 'text-gray-500' in c
                )

                if target_div:
                    size_str = target_div.get_text(strip=True)
                    size_bytes = self._parse_size_to_bytes(size_str)
                    size_bytes_str = str(size_bytes)

                    # Save to json for next time
                    try:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump({"Size": size_str, "SizeB": size_bytes_str}, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
            except Exception:
                pass

        return size_str, size_bytes_str

    def _get_model_exact_tags(self, html_file: Path, model_id: str) -> list[str]:
        """
        Load or parse model tags from model_page.html.
        Checks for model_tags.json first. If not found, parses
        model_page.html and saves the result to JSON.
        Also extracts valid tags from the model_id.
        Returns a list of tags.
        """
        json_path = html_file.parent / "model_tags.json"
        cutoff_date = datetime(2026, 7, 13, 20, 16, 0)

        if json_path.exists():
            try:
                # Check if the file was modified after the cutoff date
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

                # Find all 'a' tags with href starting with '/models?'
                for a_tag in soup.find_all('a', href=lambda h: h and h.startswith('/models?')):
                    span = a_tag.find('span')
                    if span:
                        tag_text = span.get_text(strip=True)
                        if tag_text:
                            tags.append(tag_text)

                # --- Add tags derived from model_id ---
                if model_id:
                    # Split by -, /, \, _
                    parts = re.split(r'[-/\\_]', model_id)
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue

                        # 1. Exclude tags containing a single symbol
                        if len(part) <= 1:
                            continue

                        # 2. Exclude tags containing no letters
                        if not re.search(r'[a-zA-Z]', part):
                            continue

                        part_lower = part.lower()

                        # 3. Exclude pure numbers such as 1, 2, 444, 8798 or 1.0, 2.3, 5.01 etc.
                        if re.match(r'^\d+(\.\d+)?$', part):
                            continue

                        # 4. Exclude v<number> such as v1, v2.1, v0.4 etc.
                        if re.match(r'^v\d+(\.\d+)?$', part_lower):
                            continue

                        # 5. Exclude numbers followed by B, m, or g such as 32B, 0.5B, 1.5B, 320m, 81m, 10g
                        if re.match(r'^\d+(\.\d+)?[bmg]$', part_lower):
                            continue

                        # Add if not already in tags to avoid duplicates
                        if part not in tags:
                            tags.append(part)
                # --------------------------------------

                tags = list(set([t.lower() for t in tags]))

                # Save to json for next time
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(tags, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            except Exception:
                pass

        return tags

    def _collect_existing_results(self):
        html_files = self.collect_case_files()
        if not html_files:
            return []

        existing_data = []
        tag_counts = {}  # Dictionary to count tag occurrences

        for i, html_file in enumerate(html_files):
            if i % 20 == 0:
                print(f"{i:>6} {html_file}")

            model_page_path = html_file.parent / "model_page.json"
            if not model_page_path.exists():
                continue

            # --- Load model_info.json if it exists ---
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

            # --- Load model_files_page info (Size and SizeB) ---
            size_str, size_bytes_str = self._get_model_files_info(html_file.parent)
            exact_tags = self._get_model_exact_tags(html_file, model_id)

            # --- Count tags (lowercase) ---
            for tag in exact_tags:
                tag_lower = tag.lower()
                tag_counts[tag_lower] = tag_counts.get(tag_lower, 0) + 1
            # ---------------------------------------------------

            try:
                # 1. Attempt to read and parse the JSON file
                with open(model_page_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Append the parsed data with its file path
                existing_data.append({
                    "file_path": str(model_page_path),
                    "json": data,
                    "model_info": {
                        "model_id": model_id,
                        "downloads": downloads,
                        "likes": likes
                    },
                    "size_str": size_str,
                    "size_bytes_str": size_bytes_str,
                    "exact_tags": exact_tags
                })

            except json.JSONDecodeError as e:
                # 2. Handle invalid JSON: Load as text and print
                print(f"Warning: {model_page_path} is not valid JSON. Error: {e}")
                print("Falling back to reading as raw text...")

                try:
                    with open(model_page_path, 'r', encoding='utf-8') as f:
                        raw_text = f.read()

                    print(f"\n--- Raw Content of {model_page_path} ---")
                    print(raw_text)
                    print("-" * 50 + "\n")

                    # Append a fallback dictionary so the file is still tracked in your results
                    existing_data.append({
                        "file_path": str(model_page_path),
                        "error": f"Invalid JSON: {str(e)}",
                        "raw_content": raw_text,
                        "model_info": {
                            "model_id": model_id,
                            "downloads": downloads,
                            "likes": likes
                        },
                        "size_str": size_str,
                        "size_bytes_str": size_bytes_str,
                        "exact_tags": exact_tags
                    })
                except Exception as read_err:
                    print(f"Critical Error: Could not read {model_page_path} as text. {read_err}")
                    existing_data.append({
                        "file_path": str(model_page_path),
                        "error": f"Read Error: {str(read_err)}",
                        "model_info": {
                            "model_id": model_id,
                            "downloads": downloads,
                            "likes": likes
                        },
                        "size_str": size_str,
                        "size_bytes_str": size_bytes_str,
                        "exact_tags": exact_tags
                    })

            except Exception as e:
                # Handle other potential file system errors (e.g., permissions)
                print(f"Error: An unexpected error occurred while reading {model_page_path}: {e}")
                existing_data.append({
                    "file_path": str(model_page_path),
                    "error": f"Unexpected Error: {str(e)}",
                    "model_info": {
                        "model_id": model_id,
                        "downloads": downloads,
                        "likes": likes
                    },
                    "size_str": size_str,
                    "size_bytes_str": size_bytes_str,
                    "exact_tags": exact_tags
                })

        # ==========================================
        # DETERMINE TOP TAGS
        # ==========================================
        sorted_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)
        all_top_tags = [tag for tag, count in sorted_tags[:self.TOP_TAGS_COUNT]]

        # Regex to match pattern: (word-){1,}to(-word){1,}
        # Examples: text-to-image, image-text-to-text, text-image-to-3d-image
        pattern = re.compile(r'^([a-z0-9]+-)+to(-[a-z0-9]+)+$')

        group_to = []
        group_other = []

        for tag in all_top_tags:
            if pattern.match(tag):
                group_to.append(tag)
            else:
                group_other.append(tag)

        # Sort both groups alphabetically
        group_to.sort()
        group_other.sort()

        # Combine: "to" pattern tags first, then the rest
        top_tags = group_to + group_other
        top_tags_set = set(top_tags)

        # ==========================================
        # FINAL LOOP: Print Summary as CSV & Save to File
        # ==========================================
        print(f"\nTotal records loaded: {len(existing_data)} "
              f"remaining: {len(html_files) - len(existing_data)}")
        print("=" * 120)

        csv_file_path = self.output_folder / "models_summary.csv"

        with open(csv_file_path, 'w', encoding='utf-8') as csv_file:
            # Helper function to properly escape double quotes for CSV format
            def escape_csv(val):
                """Replaces internal double quotes with two double quotes."""
                return str(val).replace('"', '""')

            def get_int_token(val):
                """Safely convert a value to an integer string, or return empty string."""
                if val is None:
                    return ""
                try:
                    return str(int(val))
                except (ValueError, TypeError):
                    return ""

            base_header = 'row_num,file_path,model_url,model_id,Size,input_modalities,Text_I,Image_I,Audio_I,Video_I,output_modalities,Text_O,Image_O,Audio_O,Video_O,3D_O,model_size,input_tokens,output_tokens,downloads,likes,SizeB'
            header = base_header + ',' + ','.join(top_tags) + ',RemainingTags'

            print(header)
            csv_file.write(header + '\n')

            for row_num, item in enumerate(existing_data, start=1):
                file_path = item.get("file_path", "Unknown Path")
                model_info_data = item.get("model_info", {})

                # Extract newly added size fields
                size_str = item.get("size_str", "")
                size_bytes_str = item.get("size_bytes_str", "")
                exact_tags = item.get("exact_tags", [])
                exact_tags_lower = [t.lower() for t in exact_tags]

                # Extract model_id, downloads, and likes from the loaded model_info
                model_id = model_info_data.get("model_id", "") or ""
                downloads = model_info_data.get("downloads", 0) or 0
                likes = model_info_data.get("likes", 0) or 0

                # Use model_id to construct the URL
                model_url = f"https://huggingface.co/{escape_csv(model_id)}" if model_id else ""

                # If an error occurred during parsing/reading, print the error and skip to next
                if "error" in item:
                    err_msg = escape_csv(item['error'])
                    empty_cols = ",".join(['""'] * 12)
                    base_row = f'{row_num},"{escape_csv(file_path)}","{model_url}","{escape_csv(model_id)}","{escape_csv(size_str)}","ERROR","{err_msg}",{empty_cols},{downloads},{likes},{size_bytes_str}'

                    empty_tag_cols = "," + ",".join([""] * len(top_tags))
                    row = base_row + empty_tag_cols + ',\"\"'

                    print(row)
                    csv_file.write(row + '\n')
                    continue

                # Extract JSON data
                data = item.get("json", {})

                # Extract fields, defaulting to empty string if missing or null (None)
                model_name = str(data["model_name"]) if data.get("model_name") is not None else ""
                model_size = str(data["model_size"]) if data.get("model_size") is not None else ""

                # Format modalities as comma-separated strings
                input_mods = data.get("input_modalities") or []
                output_mods = data.get("output_modalities") or []
                input_modalities = ",".join(str(m) for m in input_mods) if isinstance(input_mods, list) else ""
                output_modalities = ",".join(str(m) for m in output_mods) if isinstance(output_mods, list) else ""

                # Modality flags for Input
                text_i = "1" if "Text" in input_mods else ""
                image_i = "1" if "Image" in input_mods else ""
                audio_i = "1" if "Audio" in input_mods else ""
                video_i = "1" if "Video" in input_mods else ""

                # Modality flags for Output
                text_o = "1" if "Text" in output_mods else ""
                image_o = "1" if "Image" in output_mods else ""
                audio_o = "1" if "Audio" in output_mods else ""
                video_o = "1" if "Video" in output_mods else ""
                three_d_o = "1" if "3D" in output_mods else ""

                # Token counts
                input_tokens = get_int_token(data.get("input_tokens"))
                output_tokens = get_int_token(data.get("output_tokens"))

                base_row = f'{row_num},"{escape_csv(file_path)}","{model_url}","{escape_csv(model_id)}","{escape_csv(size_str)}",' \
                           f'"{escape_csv(input_modalities)}","{text_i}","{image_i}","{audio_i}","{video_i}",' \
                           f'"{escape_csv(output_modalities)}","{text_o}","{image_o}","{audio_o}","{video_o}","{three_d_o}",' \
                           f'"{escape_csv(model_size)}","{input_tokens}","{output_tokens}",{downloads},{likes},{size_bytes_str}'

                # Add top tags columns
                tag_cols = ["1" if t in exact_tags_lower else "" for t in top_tags]
                row = base_row + "," + ",".join(tag_cols)

                # Add remaining tags
                remaining_tags = [t for t in exact_tags if t.lower() not in top_tags_set]
                row += "," + escape_csv("|".join(remaining_tags))

                # print(row)
                csv_file.write(row + '\n')

        print("=" * 120)
        print(f"Processing complete. CSV saved to {csv_file_path}")

        # ==========================================
        # PRINT TAG STATISTICS
        # ==========================================
        if tag_counts:
            print("\n🏷️ TAG STATISTICS (Sorted by frequency)")
            print("-" * 40)
            sorted_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)
            for i, (tag, count) in enumerate(sorted_tags):
                print(f"{i:>6} {tag}: {count}")
        else:
            print("\n🏷️ No tags found.")
            print("\n🏷️ tags printed.")

    def collect_case_files(self) -> list[Path]:
        html_files = sorted(self.folder_path.rglob("model_page.html"))
        return html_files

    def _load_html_cases(self) -> list:
        """Scan for .html files recursively and attach prompt to each."""
        if not self.prompt_content:
            return []

        html_files = self.collect_case_files()
        if not html_files:
            return []

        cases = []
        for i, html_file in enumerate(html_files):
            model_page_path = html_file.parent / "model_page.json"
            model_page_path_txt = html_file.parent / "model_page.txt"

            if model_page_path_txt.exists():
                print(rf'skipping {html_file.parent}\{html_file.name} (previous error)')
                continue

            if model_page_path.exists():
                try:
                    target_date = datetime(2026, 6, 20, 14, 0, 0)
                    last_modified_time = datetime.fromtimestamp(Path(model_page_path).stat().st_mtime)
                    is_older = last_modified_time < target_date

                    if not is_older:
                        with open(model_page_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if "model_name" in data:
                            print(rf'skipping {html_file.parent}\{html_file.name}')
                            continue
                except Exception as e:
                    print(e)

            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()

            original_len = len(html_content)
            if HtmlCasesLoaded.USE_MARKUP_STRIPPING:
                html_content = self.html_to_formatted_text(html_content)
            else:
                html_content = self.clean_html(html_content, html_file)

            stripped_len = len(html_content)
            print(f"{i:>6} {html_file} stripped: {original_len} -> {stripped_len}")

            if stripped_len > 240000:
                print(stripped_len)

            # Try to read model_info.json from the same folder
            downloads = 0
            likes = 0
            model_info_path = html_file.parent / "model_info.json"
            if model_info_path.exists():
                try:
                    with open(model_info_path, 'r', encoding='utf-8') as f:
                        model_info = json.load(f)
                    downloads = model_info.get("Downloads", 0)
                    likes = model_info.get("Likes", 0)
                except exception as e:
                    downloads = 0
                    likes = 0

            cases.append({
                "name": html_file.stem,
                "prompt": f"{self.prompt_content}\n{html_content}\n",
                "path": html_file,
                "downloads": downloads,
                "likes": likes
            })

        # Sort by Downloads + Likes * 200 DESC
        cases.sort(key=lambda x: x["downloads"] + x["likes"] * 200, reverse=True)
        # cases.sort(key=lambda x: len(x["prompt"]))

        return cases

    def save_test_case_result(self, case_index: int, success: bool, output_text: str,
                              time_taken: float, error_msg: str = "",
                              prompt_text: str = "", input_script_len: int = 0,
                              output_script_len: int = 0, model_max_tokens: str = "",
                              model_name: str = ""):
        """
        Saves the resulting file (model output) as .json and a log file
        in the same folder as the original HTML file.
        """
        if 0 <= case_index < len(self.test_cases_data):
            case_data = self.test_cases_data[case_index]
        else:
            case_data = {"name": f"case_{case_index}", "path": None}

        basename = case_data.get("name", f"case_{case_index}")
        original_path = case_data.get("path")

        # Determine target directory: same folder as the original HTML file
        target_dir = original_path.parent if original_path else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save result as JSON or text
        file_path = target_dir / f"{basename}.json"
        try:
            if not output_text:
                raise ValueError("Empty output text")
            parsed_data = json.loads(output_text)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"⚠ JSON validation failed for {file_path}: {e}. Saving as text.")
            file_path = target_dir / f"{basename}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(output_text if output_text else "")

        # 2. Save log file
        log_file = target_dir / f"{basename}.log"
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

        # Store for potential combined outputs later
        self.results_data[basename] = {
            "log_content": log_content,
            "output_text": output_text if output_text else "",
            "success": success,
            "time_taken": time_taken,
            "prompt_length": len(prompt_text),
            "input_script_len": input_script_len,
            "output_script_len": output_script_len,
            "original_path": str(original_path)
        }

    def clean_html(self, html_content, html_file):
        """
        Clean HTML content by removing specific tags, attributes, and comments.
        1) Remove <head>, <script>, <svg> tags with their content.
        2) Remove style, class, id, xmlns, rel attributes from any tags.
        3) Remove all HTML comments (e.g., <!-- comment -->).
        4) Remove href attributes unless they contain "huggingface" or "github".
        """
        original_len = len(html_content)
        soup = BeautifulSoup(html_content, 'html.parser')

        # 1) Remove <head>, <script>, <svg> tags and their content
        for tag_name in ['head', 'script', 'svg', 'path', 'defs']:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 2) Remove style, class, id, xmlns, rel attributes from all tags
        attrs_to_remove = ['style', 'class', 'id', 'xmlns', 'rel']
        for tag in soup.find_all(True):
            for attr in attrs_to_remove:
                if tag.has_attr(attr):
                    del tag[attr]

        # 3) Remove HTML comments (including malformed ones like <!--[-1-->)
        # BeautifulSoup parses <!-- ... --> as Comment nodes
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 4) Remove href attributes unless they contain "huggingface" or "github"
        for tag in soup.find_all(href=True):
            href_val = tag['href']
            if 'huggingface' not in href_val and 'github' not in href_val:
                del tag['href']

        html_content_stripped = str(soup)
        stripped_len = len(html_content_stripped)
        # print(f"{html_file} stripped: {original_len} -> {stripped_len}")
        # if stripped_len > 240000:
        #    print(stripped_len)

        return html_content_stripped

    def html_to_formatted_text(self, html_content: str) -> str:
        """
        Strips HTML tags, removes pictures, and retains text with wiki-style
        formatting for headers and structured tables.
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # 1. Remove pictures, scripts, and styles (invisible content)
        for tag in soup.find_all(['img', 'script', 'style', 'noscript', 'svg']):
            tag.decompose()

        # 2. Remove explicitly hidden elements (e.g., display: none)
        for tag in soup.find_all(style=re.compile(r'display\s*:\s*none', re.I)):
            tag.decompose()
        for tag in soup.find_all(hidden=True):
            tag.decompose()

        # 3. Convert Tables to marked text format
        # We process tables and format them with pipes for clear readability
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                # Extract text from th and td, replace internal pipes to avoid breaking the format
                cells = [
                    td.get_text(separator=' ', strip=True).replace('|', '/')
                    for td in tr.find_all(['th', 'td'])
                ]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")

            table_text = "\n".join(rows)
            # Replace the HTML table with our formatted text block
            table.replace_with(NavigableString(f"\n[Table Start]\n{table_text}\n[Table End]\n"))

        # 4. Convert Headers (h1-h6) to Wiki format (= Title =, == Subtitle ==)
        for i in range(1, 7):
            for header in soup.find_all(f'h{i}'):
                marker = '=' * i
                text = header.get_text(strip=True)
                # Add newlines to ensure the header sits on its own line
                header.replace_with(NavigableString(f"\n{marker} {text} {marker}\n"))

        # 5. Format Lists (optional but keeps structure)
        for li in soup.find_all('li'):
            li.insert(0, NavigableString("* "))

        # 6. Extract all visible text
        # separator='\n' ensures block elements (like <p> and <div>) don't mash together
        text = soup.get_text(separator='\n')

        # 7. Clean up excessive blank lines
        text = re.sub(r'\n{3,}', '\n', text)

        return text.strip()