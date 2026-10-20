#!/usr/bin/env python
"""Convert HTML to PDF using WeasyPrint."""
import sys
from pathlib import Path

try:
    from weasyprint import HTML
except ImportError:
    print("WeasyPrint not installed. Install with: pip install weasyprint")
    sys.exit(1)

def html_to_pdf(html_file, pdf_file):
    """Convert HTML file to PDF."""
    HTML(filename=html_file).write_pdf(pdf_file)
    print(f"Converted {html_file} -> {pdf_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python html_to_pdf.py <html_file> [output_pdf]")
        sys.exit(1)

    html_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else html_file.replace(".html", ".pdf")

    html_to_pdf(html_file, pdf_file)
