#!/usr/bin/env python
"""Convert markdown to HTML with styling for PDF export."""
import sys
from pathlib import Path
import markdown

def md_to_html(md_file, html_file):
    """Convert markdown file to HTML."""
    md_content = Path(md_file).read_text(encoding="utf-8")

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists']
    )

    # Wrap with HTML template
    html_template = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 2 Report</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid #2c3e50;
            margin: 20px 0;
            padding-left: 20px;
            color: #7f8c8d;
        }}
        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 5px 0;
        }}
        hr {{
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 30px 0;
        }}
        .check {{
            color: green;
            font-weight: bold;
        }}
        .pending {{
            color: orange;
            font-weight: bold;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

    Path(html_file).write_text(html_template, encoding="utf-8")
    print(f"Converted {md_file} to {html_file}")
    print(f"Open {html_file} in browser and use Print -> Save as PDF")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_html.py <markdown_file> [output_html]")
        sys.exit(1)

    md_file = sys.argv[1]
    html_file = sys.argv[2] if len(sys.argv) > 2 else md_file.replace(".md", ".html")

    md_to_html(md_file, html_file)
