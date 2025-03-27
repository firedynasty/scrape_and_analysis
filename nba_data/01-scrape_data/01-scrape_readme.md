# Chrome Content Extractor

A Python utility that extracts content from a Chrome browser tab and formats it for AI summarization.

## Overview

This tool connects to a running Chrome instance, extracts the main content from the current webpage, and copies it to your clipboard. The copied text includes:

- A request to summarize the content
- URL and page metadata (title, publication date if available)
- The main content of the webpage in clean plain text format

The extracted content is optimized for submission to AI services for analysis or summarization.

## Features

- Connects to existing Chrome browser (no new window opened)
- Intelligently extracts main content area from webpages
- Removes unnecessary elements (scripts, styling, etc.)
- Preserves useful information like links
- Formats text for optimal AI processing
- Copies result directly to clipboard
- Works with news articles, blog posts, documentation, and more

## Requirements

- Python 3.6+
- Chrome browser
- The following Python packages:
  - `selenium`
  - `beautifulsoup4`
  - `html2text`
  - `pyperclip`

## Installation

1. Install required packages:

```bash
pip install selenium beautifulsoup4 html2text pyperclip
```

2. Make sure you have Chrome installed on your system.

## Usage

1. Start Chrome with remote debugging enabled:

   **macOS:**
   ```bash
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
   ```

   **Windows:**
   ```bash
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
   ```

   **Linux:**
   ```bash
   google-chrome --remote-debugging-port=9222
   ```

2. Navigate to the webpage you want to extract in Chrome.

3. Run the script:

   ```bash
   python content_extractor.py
   ```

4. The script will extract content from the active tab and copy it to your clipboard.

5. Paste the content into your AI tool or text editor.

## How It Works

1. Connects to Chrome using the remote debugging protocol
2. Gets the HTML content of the current page
3. Uses BeautifulSoup to extract the main content area
4. Converts the HTML to clean plain text
5. Adds metadata (URL, title, date) at the top
6. Copies the formatted text to clipboard

## Customization

You can modify the content selectors in the `extract_main_content()` function to better target specific websites or content types.

## Troubleshooting

- **Cannot connect to Chrome**: Make sure Chrome is running with the `--remote-debugging-port=9222` flag
- **Content extraction issues**: Try modifying the content selectors to match the structure of the website
- **Empty clipboard**: Check the console output for error messages

## License

[Your license choice]

## Author

[Your name/organization]
