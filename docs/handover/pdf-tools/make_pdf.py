import sys, pathlib
from playwright.sync_api import sync_playwright

src = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
SCALE = 0.8
W = round((210 - 22) / 25.4 * 96 / SCALE)  # layout width the PDF will use

footer = ('<div style="width:100%;font-family:Helvetica,Arial,sans-serif;font-size:7.5px;color:#6B7383;'
          'padding:0 11mm;display:flex;justify-content:space-between">'
          '<span>Reef Vision Studio · Handover guide</span>'
          '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": W, "height": 1200})
    pg.emulate_media(media="print", color_scheme="light")
    pg.goto(src.as_uri(), wait_until="networkidle")
    pg.evaluate("document.fonts.ready")
    pg.evaluate("Promise.all([...document.images].map(i=>i.complete?1:new Promise(r=>{i.onload=i.onerror=r})))")
    bad = pg.evaluate("""(W)=>{const r=[];for(const e of document.querySelectorAll('body *')){
      const b=e.getBoundingClientRect(); if(!b.width) continue;
      const over=b.right>W+1||b.left<-1; const scroll=e.scrollWidth>e.clientWidth+2&&getComputedStyle(e).overflowX!=='visible';
      const clip=e.scrollWidth>e.clientWidth+2&&['hidden','clip'].includes(getComputedStyle(e).overflowX)&&e.tagName!=='IMG';
      if(over||scroll) r.push([e.tagName,e.className,Math.round(b.right),e.scrollWidth,e.clientWidth,(e.textContent||'').trim().slice(0,50)]);}
      return r.slice(0,40)}""", W)
    print("layout width", W, "overflowing elements:", len(bad))
    for x in bad: print("  ", x)
    pg.pdf(path=str(out), format="A4", scale=SCALE, print_background=True, prefer_css_page_size=True,
           display_header_footer=True, header_template="<span></span>", footer_template=footer,
           outline=True, tagged=True)
    b.close()
print("wrote", out)
