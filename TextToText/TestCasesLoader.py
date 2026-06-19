import os
import json
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup, Comment

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

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
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

    def _load_html_cases(self) -> list:
        """Scan for .html files recursively and attach prompt to each."""
        if not self.prompt_content:
            return []
        html_files = sorted(self.folder_path.rglob("*.html"))
        if not html_files:
            return []
        cases = []
        for html_file in html_files:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            html_content = self.clean_html(html_content)
            cases.append({
                "name": html_file.stem,
                "prompt": f"{self.prompt_content}\n{html_content}\n",
                "path": html_file
            })
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

        # 1. Save result as JSON
        json_file = target_dir / f"{basename}.json"
        result_data = {
            "test_case": basename,
            "model": model_name,
            "success": success,
            "error": error_msg if error_msg else None,
            "time_taken_seconds": time_taken,
            "output_text": output_text if output_text else ""
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

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

    def clean_html(self, html_content):
        """
        Clean HTML content by removing specific tags, attributes, and comments.
        1) Remove <head>, <script>, <svg> tags with their content.
        2) Remove style, class, id, xmlns, rel attributes from any tags.
        3) Remove all HTML comments (e.g., <!-- comment -->).
        """
        original_len = len(html_content)
        soup = BeautifulSoup(html_content, 'html.parser')

        # 1) Remove <head>, <script>, <svg> tags and their content
        for tag_name in ['head', 'script', 'svg']:
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

        html_content_stripped = str(soup)
        stripped_len = len(html_content_stripped)

        print(f"html stripped: {original_len} -> {stripped_len}")

        return html_content_stripped