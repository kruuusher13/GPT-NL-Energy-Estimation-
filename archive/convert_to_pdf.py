#!/usr/bin/env python3
"""Convert MEETING_SCRIPT.md to a clean PDF using pandoc + weasyprint."""
import weasyprint, subprocess, tempfile, os

md_path = "/Users/hornet/Projects/TNO/MEETING_SCRIPT.md"
pdf_path = "/Users/hornet/Projects/TNO/MEETING_SCRIPT.pdf"

# Step 1: pandoc md -> html
result = subprocess.run(
    ["pandoc", md_path, "-t", "html5", "--standalone"],
    capture_output=True, text=True, timeout=30
)
html = result.stdout
if result.returncode != 0:
    print(f"pandoc error: {result.stderr}")
    exit(1)

# Step 2: weasyprint html -> pdf
doc = weasyprint.HTML(string=html)
doc.write_pdf(pdf_path)

size = os.path.getsize(pdf_path)
print(f"PDF generated: {pdf_path} ({size/1024:.0f} KB)")
