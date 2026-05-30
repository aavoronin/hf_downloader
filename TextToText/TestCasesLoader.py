from pathlib import Path


class TestCasesLoaded:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.prompt_content = self._load_prompt()
        self.test_prompts = self._load_test_cases()

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
            cases.append(self.prompt_content + sql_content)
        return cases

    def get_test_prompts(self) -> list:
        """Return the list of combined prompt + SQL test cases."""
        return self.test_prompts