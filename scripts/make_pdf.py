import markdown, os
from xhtml2pdf import pisa

# Resolve paths relative to the repo root, so this runs from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))   # scripts/
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
BASE = DOCS                                          # image paths in Report.md resolve from docs/
md_text = open(os.path.join(DOCS, "Report.md"), encoding="utf-8").read()

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "sane_lists"],
)

css = """
@page { size: A4; margin: 1.8cm 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.4; color: #1a1a1a; }
h1 { font-size: 19pt; color: #14315c; border-bottom: 2px solid #14315c; padding-bottom: 4px; }
h2 { font-size: 14pt; color: #14315c; margin-top: 16px; }
h3 { font-size: 12pt; color: #2a4d7a; }
code { font-family: Courier, monospace; background: #f2f2f2; font-size: 9.5pt; }
pre { background: #f5f5f5; padding: 8px; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; }
th, td { border: 1px solid #999; padding: 1.5px 2px; text-align: left; }
th { background: #e8eef6; }
img { width: 15cm; }
hr { border: none; border-top: 1px solid #ccc; }
"""

html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html_body}</body></html>"

def link_callback(uri, rel):
    path = os.path.join(BASE, uri.replace("/", os.sep))
    return path if os.path.isfile(path) else uri

with open(os.path.join(DOCS, "Report.pdf"), "wb") as f:
    status = pisa.CreatePDF(html, dest=f, link_callback=link_callback)

print("ERROR" if status.err else "OK -> docs/Report.pdf")
