import re
import os
import requests
from bs4 import BeautifulSoup, NavigableString, Tag


def parse_alice():
    """Main entry point: process Alice in Wonderland and save OCR-ready text."""
    print("Processing Alice's Adventures in Wonderland for OCR test...")

    # Generate the OCR-ready text
    ocr_text = prepare_ocr_test()

    # Define output path
    output_dir = r"D:\Data\ocr_test\Alice"
    output_file = os.path.join(output_dir, "alice.txt")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the complete file (no preview, full content)
    with open(output_file, 'w', encoding='ascii', errors='ignore') as f:
        f.write(ocr_text)

    print(f"✓ Saved {len(ocr_text):,} characters to '{output_file}'")
    print(f"✓ All characters are ASCII-compatible for OCR processing")


def prepare_ocr_test():
    """
    Download HTML from Project Gutenberg Alice in Wonderland,
    extract text with precise line break handling:
    - <br/> and table cells -> single newline
    - <p>, <h1>-<h6>, <div.chapter> -> paragraph blocks separated by blank lines
    - Internal whitespace in paragraphs collapsed to single spaces
    Convert special characters to ASCII equivalents for OCR testing.

    Returns:
        str: Cleaned plain text with ASCII characters only
    """
    alice_url = "https://www.gutenberg.org/files/11/11-h/11-h.htm"

    # Download the HTML content with proper headers
    response = requests.get(alice_url, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    response.encoding = 'utf-8'
    html_content = response.text

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove non-content elements that would clutter the text
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'meta', 'link']):
        tag.decompose()

    # Character normalization: convert Unicode punctuation to ASCII equivalents
    def to_ascii(text):
        # Smart quotes to straight quotes
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201a', "'").replace('\u201b', "'")

        # Dashes
        text = text.replace('\u2014', '--')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2015', '--')
        text = text.replace('\u2212', '-')

        # Other substitutions
        text = text.replace('\u2026', '...')
        text = text.replace('\u00ab', '"').replace('\u00bb', '"')
        text = text.replace('\u2032', "'").replace('\u2033', '"')

        # Remove any remaining non-ASCII
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text

    # Tags that should produce a single newline after their content
    INLINE_BREAK_TAGS = {'br', 'td', 'th', 'dt', 'dd'}

    # Tags that represent block-level content (separated by blank lines)
    BLOCK_TAGS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                  'div', 'section', 'article', 'blockquote', 'pre', 'li', 'tr'}

    def extract_text_with_breaks(element, in_pre=False):
        """
        Recursively extract text preserving:
        - <br/> and <td> -> \n (single newline)
        - block elements -> \n\n (blank line separator)
        - inline text -> spaces normalized
        """
        if isinstance(element, NavigableString):
            if in_pre:
                return str(element)
            # Normalize whitespace but preserve intentional spaces
            return re.sub(r'[ \t]+', ' ', str(element))

        tag_name = element.name.lower() if isinstance(element, Tag) else ''

        # Handle <br> as explicit newline
        if tag_name == 'br':
            return '\n'

        # Handle <hr> as section break
        if tag_name == 'hr':
            return '\n' + '-' * 40 + '\n'

        # Handle <pre> - preserve formatting
        if tag_name == 'pre':
            return element.get_text()

        new_in_pre = in_pre or tag_name in {'pre', 'code'}

        # Collect content from children
        parts = []
        for child in element.children:
            child_text = extract_text_with_breaks(child, new_in_pre)
            if child_text:
                parts.append(child_text)
                # Add single newline after inline-break tags
                if tag_name in INLINE_BREAK_TAGS:
                    parts.append('\n')

        result = ''.join(parts)

        # Add blank line after block-level elements (except when inside pre)
        if not in_pre and tag_name in BLOCK_TAGS:
            result += '\n\n'

        return result

    # Find main content
    main_content = soup.body if soup.body else soup

    # Extract text with structural breaks
    raw_text = extract_text_with_breaks(main_content)

    # Post-process: normalize whitespace patterns
    # Collapse multiple spaces/tabs to single space (but not newlines)
    raw_text = re.sub(r'[ \t]+', ' ', raw_text)

    # Normalize: single newline followed by spaces -> just newline
    raw_text = re.sub(r'\n[ \t]*', '\n', raw_text)

    # Normalize paragraph breaks: 2+ newlines -> exactly 2
    raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)

    # Strip trailing whitespace from each line but preserve blank lines
    lines = [line.rstrip() for line in raw_text.split('\n')]
    raw_text = '\n'.join(lines)

    # Remove leading blank lines
    raw_text = raw_text.lstrip('\n')

    # Convert special characters to ASCII
    ascii_text = to_ascii(raw_text)

    # Final pass: ensure no trailing spaces on any line
    final_lines = [line.rstrip() for line in ascii_text.split('\n')]
    clean_text = '\n'.join(final_lines)

    # Remove any trailing whitespace at end of file
    clean_text = clean_text.rstrip() + '\n'

    return clean_text

