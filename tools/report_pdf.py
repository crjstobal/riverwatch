"""Render docs/report.md to a PDF (Markdown -> HTML -> Chromium print), images and tables included.

usage: report_pdf.py [out.pdf]      default docs/RiverWatch_technical_report.pdf
"""
import asyncio
import sys
from pathlib import Path

import markdown
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font: 10.5pt/1.45 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #14202b; }
h1 { font-size: 20pt; margin: 0 0 4pt; } h2 { font-size: 14pt; margin: 16pt 0 6pt; border-bottom: 1px solid #dfe5ea; padding-bottom: 3pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6pt 0; page-break-inside: avoid; }
th, td { border: 1px solid #dfe5ea; padding: 3pt 5pt; vertical-align: top; text-align: left; } th { background: #f4f6f8; }
code { font: 9pt ui-monospace, Menlo, monospace; background: #f4f6f8; padding: 0 2pt; border-radius: 2pt; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
p, li { orphans: 3; widows: 3; }
"""


async def main(out):
    md = (ROOT / "docs/report.md").read_text()
    html = markdown.markdown(md, extensions=["tables", "fenced_code"])
    page_html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    tmp = ROOT / "docs" / "_report.html"  # relative image paths resolve from docs/
    tmp.write_text(page_html)
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.goto(tmp.as_uri(), wait_until="networkidle")
        await pg.pdf(path=str(out), format="A4", print_background=True,
                     margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"},
                     display_header_footer=True, header_template="<span></span>",
                     footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'>RiverWatch technical report · <span class='pageNumber'></span>/<span class='totalPages'></span></div>")
        await b.close()
    tmp.unlink()
    print(out)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/RiverWatch_technical_report.pdf"))
