import requests
from bs4 import BeautifulSoup, NavigableString, Comment
import pandas as pd
import re


class HFModelLister:
    def __init__(self, base_url, token=None):
        self.print_html = False
        self.base_url = base_url
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.results = []

    @staticmethod
    def parse_number(text):
        """Safely parse numbers with k/M/B suffixes, return 0 if invalid."""
        if not text:
            return 0
        text = str(text).replace(",", "").strip()
        if not text:
            return 0
        # Filter out years (1990-2099)
        if text.isdigit() and 1990 <= int(text) <= 2099:
            return 0
        try:
            if text.lower().endswith("k"):
                return int(float(text[:-1]) * 1_000)
            elif text.lower().endswith("m"):
                return int(float(text[:-1]) * 1_000_000)
            elif text.lower().endswith("b"):
                return int(float(text[:-1]) * 1_000_000_000)
            elif text.isdigit():
                return int(text)
            # Try extracting leading numeric part
            match = re.match(r'^([\d.]+)', text)
            if match:
                return int(float(match.group(1)))
        except (ValueError, TypeError):
            pass
        return 0

    def _extract_purpose(self, metrics_div):
        """Extract purpose text from the metrics div - text after first SVG."""
        if not metrics_div:
            return ""

        # Find the first SVG (purpose/task icon)
        first_svg = metrics_div.find("svg")
        if not first_svg:
            return ""

        # Get all text content from the metrics div
        all_text = metrics_div.get_text(" ", strip=True)

        # The purpose is typically the first meaningful text after the SVG
        # Split by bullet separator and take the first part
        parts = all_text.split("•")
        if parts:
            purpose = parts[0].strip()
            # Filter out numbers, dates, and "Updated" label
            if purpose and not re.match(r'^[\d.]+[kKmMbB]?$', purpose) and purpose.lower() not in ['updated', '']:
                return purpose

        return ""

    def _extract_metric_after_svg(self, container, path_startswith):
        """Find SVG with matching path 'd' attribute and return next text number."""
        for svg in container.find_all("svg"):
            path = svg.find("path")
            if path and path.get("d", "").startswith(path_startswith):
                # Get next sibling that contains the number
                next_sibling = svg.next_sibling
                while next_sibling:
                    if isinstance(next_sibling, NavigableString):
                        text = str(next_sibling).strip()
                        if text and re.match(r'^[\d.]+[kKmMbB]?$', text):
                            return text
                    elif hasattr(next_sibling, 'get_text'):
                        text = next_sibling.get_text(strip=True)
                        if text and re.match(r'^[\d.]+[kKmMbB]?$', text):
                            return text
                    next_sibling = next_sibling.next_sibling
                break
        return None

    def parse_page(self, url):
        resp = self.session.get(url)
        if resp.status_code != 200:
            print(f"Failed to load {url}, status {resp.status_code}")
            return False

        soup = BeautifulSoup(resp.text, "html.parser")
        model_cards = soup.select("a.flex.items-center.justify-between")
        if not model_cards:
            return False

        for i, card in enumerate(model_cards, start=1):
            try:
                # Debug: print HTML for every 5th card
                if self.print_html and i % 5 == 0:
                    print(f"\n--- Inner HTML of card #{i} ---")
                    print(card.decode_contents())

                # Model ID
                header = card.select_one("header[title]")
                model_id = header["title"].strip() if header and header.get("title") else ""

                # Purpose: extract from HTML structure
                metrics_div = card.select_one("div.text-sm.leading-tight.text-gray-400")
                model_purpose = self._extract_purpose(metrics_div) if metrics_div else ""

                # Updated date
                time_tag = card.find("time")
                update_date = time_tag.get_text(strip=True) if time_tag else ""

                # Downloads & Likes: target specific SVG icons
                downloads = likes = 0

                if metrics_div:
                    # Download icon: path starts with "M26 24v4H6v-4"
                    dl_text = self._extract_metric_after_svg(metrics_div, "M26 24v4H6v-4")
                    downloads = self.parse_number(dl_text) if dl_text else 0

                    # Like (heart) icon: path starts with "M22.45,6"
                    like_text = self._extract_metric_after_svg(metrics_div, "M22.45,6")
                    likes = self.parse_number(like_text) if like_text else 0

                self.results.append({
                    "Model ID": model_id,
                    "Purpose": model_purpose,
                    "Updated": update_date,
                    "Downloads": downloads,
                    "Likes": likes
                })
            except Exception as e:
                print(f"Error parsing card: {e}")
        return True

    def fetch_all_pages(self):
        page = 0
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
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', 300)
        pd.set_option('display.width', 300)
        pd.set_option('display.max_rows', 200)
        print(df)