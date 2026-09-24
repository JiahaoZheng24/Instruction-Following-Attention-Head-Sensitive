"""Export the LAST slide of a figure deck to PNG (preview) and a one-page PDF (LaTeX), with PowerPoint.
  python ICLR/figure/code/render.py ICLR/figure/fig1_mechanism.pptx
writes ICLR/figure/fig1_mechanism_preview.png and ICLR/ICLR_quantization/figures/fig1_mechanism.pdf
(an explicit second argument overrides the PDF path). Needs pywin32 and pymupdf (base python has both).
"""
import os
import sys

import fitz  # pymupdf
import win32com.client

src = os.path.abspath(sys.argv[1])
stem = os.path.splitext(os.path.basename(src))[0]
root = os.path.abspath(os.path.join(os.path.dirname(src), '..'))            # ICLR/
pdf = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(root, 'ICLR_quantization', 'figures', stem + '.pdf')
png = os.path.join(os.path.dirname(src), stem + '_preview.png')
tmp = pdf + '.all.pdf'

app = win32com.client.Dispatch('PowerPoint.Application')
pres = app.Presentations.Open(src, WithWindow=False)
try:
    n = pres.Slides.Count
    w_cm = pres.PageSetup.SlideWidth / 28.35
    pres.Slides(n).Export(png, 'PNG', int(w_cm * 180), 0)     # ~450 dpi
    pres.SaveAs(tmp, 32)
finally:
    pres.Close()
    app.Quit()
doc = fitz.open(tmp)
out = fitz.open()
out.insert_pdf(doc, from_page=n - 1, to_page=n - 1)
out.save(pdf)
out.close()
doc.close()
os.remove(tmp)
print(f'slides {n}; png {png}; pdf {pdf} (last slide only)')
