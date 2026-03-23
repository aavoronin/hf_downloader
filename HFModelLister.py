import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

class HFModelLister:
    def __init__(self, base_url, token=None):
        self.base_url = base_url
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.results = []

    @staticmethod
    def parse_number(text):
        text = text.replace(",", "").strip()
        if text.endswith("k"):
            return int(float(text[:-1]) * 1_000)
        elif text.endswith("M"):
            return int(float(text[:-1]) * 1_000_000)
        elif text.endswith("B"):
            return int(float(text[:-1]) * 1_000_000_000)
        elif text.isdigit():
            return int(text)
        return 0

    def parse_page(self, url):
        resp = self.session.get(url)
        if resp.status_code != 200:
            print(f"Failed to load {url}, status {resp.status_code}")
            return False

        soup = BeautifulSoup(resp.text, "html.parser")
        model_cards = soup.select("a.flex.items-center.justify-between")
        if not model_cards:
            return False

        for card in model_cards:
            try:
                # Model ID
                header = card.select_one("header[title]")
                model_id = header["title"].strip() if header else ""

                # Purpose
                purpose_tag = card.select_one("div.text-sm.leading-tight.text-gray-400")
                purpose_text = purpose_tag.get_text(" ", strip=True) if purpose_tag else ""
                model_purpose = ""
                if "Automatic Speech Recognition" in purpose_text:
                    model_purpose = "Automatic Speech Recognition"
                elif "Text-to-Speech" in purpose_text:
                    model_purpose = "Text-to-Speech"

                # Updated
                time_tag = card.find("time")
                update_date = time_tag.get_text(strip=True) if time_tag else ""

                # Downloads / Likes
                downloads = likes = 0
                if purpose_tag:
                    # regex for numbers like 8.84k, 0.2B, 378M
                    numbers = re.findall(r'\d+\.?\d*[kMB]?', purpose_text)
                    if numbers:
                        if len(numbers) >= 2:
                            downloads, likes = self.parse_number(numbers[-2]), self.parse_number(numbers[-1])
                        elif len(numbers) == 1:
                            likes = self.parse_number(numbers[-1])

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
        page = 0  # start with page=0 (or first page has no p)
        while True:
            url = self.base_url if page == 0 else f"{self.base_url}&p={page}"
            print(f"Fetching page {page} -> {url}")
            has_results = self.parse_page(url)
            if not has_results:
                break
            page += 1

    def show_results(self):
        if not self.results:
            print("No results collected")
            return
        df = pd.DataFrame(self.results)
        # Pandas display options
        pd.set_option('display.max_columns', None)  # show all columns
        pd.set_option('display.max_colwidth', 300)  # widen columns
        pd.set_option('display.width', 300)  # total table width
        pd.set_option('display.max_rows', 200)  # show up to 200 rows (adjust as needed)
        print(df)