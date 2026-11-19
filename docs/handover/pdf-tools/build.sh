#!/bin/zsh
# Rebuild site/Reef-Vision-Handover-Guide.pdf from site/index.html.
# Needs: python -m pip install playwright pymupdf pillow && python -m playwright install chromium
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
rm -rf site/_printimg && mkdir site/_printimg
$PY - <<'PYEOF'
import glob, os
from PIL import Image
src = open("site/index.html", encoding="utf-8").read()
for f in glob.glob("site/img/*.webp"):          # JPEG copies keep the PDF small
    im = Image.open(f); name = os.path.basename(f)[:-5]
    if im.mode in ("RGBA", "LA") and im.getchannel("A").getextrema()[0] < 250:
        continue
    im.convert("RGB").save(f"site/_printimg/{name}.jpg", quality=84, optimize=True)
    src = src.replace(f'src="img/{name}.webp"', f'src="_printimg/{name}.jpg"')
open("site/_print.html", "w", encoding="utf-8").write('<!doctype html>\n<html lang="en"><meta charset="utf-8">\n' + src)
PYEOF
$PY pdf-tools/make_pdf.py site/_print.html site/Reef-Vision-Handover-Guide.pdf
rm -rf site/_print.html site/_printimg
$PY pdf-tools/post_pdf.py site/Reef-Vision-Handover-Guide.pdf site/index.html
