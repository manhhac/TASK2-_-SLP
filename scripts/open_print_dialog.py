#!/usr/bin/env python
"""Open HTML file in browser with print dialog."""
import webbrowser
from pathlib import Path

def open_with_print(html_file):
    """Open HTML file in browser (user can then Ctrl+P to print)."""
    abs_path = Path(html_file).resolve()
    webbrowser.open(f"file:///{abs_path}")
    print(f"Opened {html_file} in browser")
    print("Press Ctrl+P to open print dialog, then 'Save as PDF'")

if __name__ == "__main__":
    html_file = "docs/TASK2_REPORT.html"
    open_with_print(html_file)
