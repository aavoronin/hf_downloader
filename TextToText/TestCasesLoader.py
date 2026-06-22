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

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
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
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store results for combined output files
        self.results_data = {}

    def _collect_existing_results(self):
        html_files = self.collect_case_files()
        if not html_files:
            return []

        existing_data = []
        for i, html_file in enumerate(html_files):
            model_page_path = html_file.parent / "model_page.json"
            if not model_page_path.exists():
                continue

            try:
                # 1. Attempt to read and parse the JSON file
                with open(model_page_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Append the parsed data with its file path
                existing_data.append({
                    "file_path": str(model_page_path),
                    "json": data
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
                        "raw_content": raw_text
                    })
                except Exception as read_err:
                    print(f"Critical Error: Could not read {model_page_path} as text. {read_err}")
                    existing_data.append({
                        "file_path": str(model_page_path),
                        "error": f"Read Error: {str(read_err)}",
                    })

            except Exception as e:
                # Handle other potential file system errors (e.g., permissions)
                print(f"Error: An unexpected error occurred while reading {model_page_path}: {e}")
                existing_data.append({
                    "file_path": str(model_page_path),
                    "error": f"Unexpected Error: {str(e)}",
                })

        # ==========================================
        # FINAL LOOP: Print Summary as CSV
        # ==========================================
        print(f"\nTotal records loaded: {len(existing_data)}")
        print("=" * 120)

        # Helper function to properly escape double quotes for CSV format
        def escape_csv(val):
            """Replaces internal double quotes with two double quotes."""
            return str(val).replace('"', '""')

        for row_num, item in enumerate(existing_data, start=1):
            file_path = item.get("file_path", "Unknown Path")

            # If an error occurred during parsing/reading, print the error and skip to next
            if "error" in item:
                err_msg = escape_csv(item['error'])
                # Kept the same 6-column structure so CSV parsers don't break on error rows
                print(f'{row_num},"{escape_csv(file_path)}","ERROR","{err_msg}","",""')
                continue

            # Extract JSON data
            data = item.get("json", {})

            # Extract fields, defaulting to empty string if missing or null (None)
            model_name = str(data["model_name"]) if data.get("model_name") is not None else ""
            model_size = str(data["model_size"]) if data.get("model_size") is not None else ""

            # Format modalities as comma-separated strings (e.g., "Text,Image,Audio")
            input_mods = data.get("input_modalities")
            input_modalities = ",".join(str(m) for m in input_mods) if isinstance(input_mods, list) else ""

            output_mods = data.get("output_modalities")
            output_modalities = ",".join(str(m) for m in output_mods) if isinstance(output_mods, list) else ""

            if row_num == 1:
                print('row_num,file,model_name,input_modalities,output_modalities,model_size')

            # Print the formatted row with properly escaped symbols for CSV
            print(f'{row_num},"{escape_csv(file_path)}","{escape_csv(model_name)}",'
                  f'"{escape_csv(input_modalities)}","{escape_csv(output_modalities)}","{escape_csv(model_size)}"')

        print("=" * 120)
        print("Processing complete.")

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
        #cases.sort(key=lambda x: len(x["prompt"]))
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
        try:
            if not output_text:
                raise ValueError("Empty output text")
            parsed_data = json.loads(output_text)
            file_path = target_dir / f"{basename}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"⚠ JSON validation failed for {basename}: {e}. Saving as text.")
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

        #print(f"{html_file} stripped: {original_len} -> {stripped_len}")

        #if stripped_len > 240000:
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
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()


