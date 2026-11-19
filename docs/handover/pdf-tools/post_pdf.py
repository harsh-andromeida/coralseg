import sys, re, html, pymupdf
pdf, src = sys.argv[1], sys.argv[2]
d = pymupdf.open(pdf)
names = d.resolve_names()  # id -> {'page': n, 'to': (x, y), ...}
s = open(src, encoding="utf-8").read()
# outline from the contents section (chapters + sections), plus figures
toc = [[1, "Cover", 1], [1, "Contents", names["contents"]["page"] + 1]]
cont = s[s.index('<section class="contents"'):s.index('</section>', s.index('<section class="contents"'))]
for blk in re.finditer(r'<div class="cg k\d"><a class="cg-h" href="#([^"]+)"><span>(\d+)</span>(.*?)</a><ol>(.*?)</ol></div>', cont, re.S):
    cid, num, title, body = blk.groups()
    toc.append([1, f"{num}  {html.unescape(title)}", names[cid]["page"] + 1])
    for li in re.finditer(r'<li( class="lv3")?><a href="#([^"]+)">(?:<span>([^<]+)</span>)?(.*?)</a></li>', body):
        lvl = 3 if li.group(1) else 2
        label = (li.group(3) + "  " if li.group(3) else "") + html.unescape(li.group(4))
        toc.append([lvl, label, names[li.group(2)]["page"] + 1])
toc.append([1, "Figures", names["contents"]["page"] + 1])
for f in re.finditer(r'<li><a href="#(fig-\d-\d)"><span>([^<]+)</span>(.*?)</a></li>', cont):
    toc.append([2, f"Figure {f.group(2)}  {html.unescape(f.group(3))}", names[f.group(1)]["page"] + 1])
d.set_toc(toc)
d.set_metadata({"title": "Reef Vision Studio · Handover Guide", "author": "Andromeida", "subject": "Project handover documentation for Reef Vision Studio (CoralSeg)"})

out = pdf.replace(".pdf", ".tmp.pdf")
d.save(out, garbage=1, deflate=True)
d.close()
import os; os.replace(out, pdf)
print("outline entries", len(toc))
