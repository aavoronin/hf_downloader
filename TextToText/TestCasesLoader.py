import os
from pathlib import Path
from datetime import datetime


class TestCasesLoaded:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.prompt_content = self._load_prompt()
        self.test_prompts = self._load_test_cases()

        # Create output folder structure
        # out_folder = f"out/folder_path" -> e.g. out/TestCases/Oracle/Basic
        out_folder = Path("out") / self.folder_path

        # Create timestamp folder YYYYMMDDHHMMSS
        # YYYYMMDDHHMMSS is remembered when the group of test cases is started
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = out_folder / timestamp

        # Create the directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        self.filenames = []  # Store stems to map results back to filenames
        for sql_file in sql_files:
            self.filenames.append(sql_file.stem)  # e.g. "test_case1"
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            cases.append(f"{self.prompt_content}\n====SCRIPT START====\n"
                         f"{sql_content}\n"
                         f"====SCRIPT START====")
        return cases

    def get_test_prompts(self) -> list:
        """Return the list of combined prompt + SQL test cases."""
        return self.test_prompts

    def save_test_case_result(self, case_index: int, success: bool, output_text: str,
                              time_taken: float, error_msg: str = "",
                              prompt_text: str = "", input_script_len: int = 0,
                              output_script_len: int = 0, model_max_tokens: str = ""):
        """
        Saves the 3 result files for a specific test case into the timestamped output directory.
        """
        if 0 <= case_index < len(self.filenames):
            basename = self.filenames[case_index]
        else:
            basename = f"case_{case_index}"

        # 1. tast_case.sql (Result) -> basename.sql
        sql_file = self.output_dir / f"{basename}.sql"
        with open(sql_file, 'w', encoding='utf-8') as f:
            f.write(output_text if output_text else "")

        # 2. tast_case_PROMPT.txt (Full prompt) -> basename_PROMPT.txt
        prompt_file = self.output_dir / f"{basename}_PROMPT.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_text)

        # 3. tast_case.log -> basename.log
        log_file = self.output_dir / f"{basename}.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Test Case: {basename}\n")
            f.write(f"Status: {'Success' if success else 'Failure'}\n")
            f.write(f"Error: {error_msg if error_msg else 'None'}\n")
            f.write(f"Time Taken: {time_taken:.4f}s\n")
            f.write(f"Prompt Length: {len(prompt_text)}\n")
            f.write(f"Input Tokens (Approx): {len(prompt_text)}\n")
            f.write(f"Model Max Input/Output Tokens: {model_max_tokens}\n")
            f.write(f"Length of Input Script: {input_script_len}\n")
            f.write(f"Length of Output Script: {output_script_len}\n")