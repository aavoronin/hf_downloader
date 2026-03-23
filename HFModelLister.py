import requests
from bs4 import BeautifulSoup
import pandas as pd

class HFModelLister:
    def __init__(self, base_url):
        """
        Initialize with the starting URL (Hugging Face models search page).
        Example:
        "https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=likes&search=rus"
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []

    def parse_page(self, url):
        """Parse a single Hugging Face models page and extract data"""
        resp = self.session.get(url)
        if resp.status_code != 200:
            print(f"Failed to load {url}, status {resp.status_code}")
            return False

        soup = BeautifulSoup(resp.text, "html.parser")
        model_cards = soup.find_all("div", class_="w-full truncate")
        if not model_cards:
            return False  # stop when no results

        for card in model_cards:
            try:
                # Initialize all variables
                model_id = ""
                model_purpose = ""
                update_date = ""
                downloads = ""
                likes = ""

                # Model ID
                header = card.find("header", {"title": True})
                if header:
                    model_id = header["title"].strip()

                # Purpose and other metadata
                meta_div = card.find("div", class_="flex items-center overflow-hidden")
                if meta_div:
                    text_parts = meta_div.get_text(separator="|").split("|")
                    for part in text_parts:
                        part = part.strip()
                        if not part:
                            continue
                        if "Updated" in part:
                            update_date = part.replace("Updated", "").strip()
                        elif "k" in part or "M" in part:
                            if not downloads:
                                downloads = part
                        elif part.isdigit():
                            likes = part
                        elif "Automatic Speech Recognition" in part or "Text-to-Speech" in part:
                            model_purpose = part

                self.results.append({
                    "Model ID": model_id,
                    "Purpose": model_purpose,
                    "Updated": update_date,
                    "Downloads": downloads,
                    "Likes": likes
                })
            except Exception as e:
                print("Error parsing card:", e)

        return True

    def fetch_all_pages(self):
        """Fetch pages until no results"""
        page = 1
        while True:
            if "p=" in self.base_url:
                url = self.base_url.split("&p=")[0] + f"&p={page}"
            else:
                url = self.base_url + f"&p={page}"
            print(f"Fetching page {page} -> {url}")
            has_results = self.parse_page(url)
            if not has_results:
                break
            page += 1

    import pandas as pd

    def show_results(self):
        """Print results as a wide table"""
        if not self.results:
            print("No results collected")
            return

        df = pd.DataFrame(self.results)

        # Set Pandas display options
        pd.set_option('display.max_columns', None)  # show all columns
        pd.set_option('display.max_colwidth', 300)  # increase column width
        pd.set_option('display.width', 300)  # total table width

        print(df)