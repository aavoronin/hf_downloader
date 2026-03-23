# load_token.py
def load_hf_token(file_path=r"D:\AIs\token.txt"):
    """Read HuggingFace token from a file"""
    try:
        with open(file_path, "r") as f:
            token = f.read().strip()
            return token
    except FileNotFoundError:
        print(f"Token file {file_path} not found!")
        return None

HF_TOKEN = load_hf_token()