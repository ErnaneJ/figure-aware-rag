"""Check each caption extracted at ingestion against the figure it belongs to.

Reads the benchmark PDF (gitignored, download via DOI 10.1190/1.2098670) and the ingestion log;
writes caption_check.csv. No API calls. The reference figure of an image is the "Figure N." caption
block printed directly below it in the same column. Images are enumerated in the same order as
indexing/processor.py (page.get_images), so img_doc_index matches the ingestion log."""
import re
import sys
from pathlib import Path

import fitz
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "study2"))
from rebuild_records import parse_log  # noqa: E402

PDF = HERE.parent / "gsgpy_70_5_3SO.pdf"
FIG = re.compile(r"\s*Figure (\d+)\b")


def reference_figure(img_rect, captions):
    """Nearest caption below the image whose column overlaps it horizontally."""
    best = None
    for rect, num in captions:
        overlap = min(img_rect.x1, rect.x1) - max(img_rect.x0, rect.x0)
        gap = rect.y0 - img_rect.y1
        if overlap > 0 and -5 <= gap <= 40 and (best is None or gap < best[0]):
            best = (gap, num)
    return best[1] if best else None


log = parse_log()
by_index = {v["img_doc_index"]: v for v in log.values()}
rows, n = [], 0
for page_num, page in enumerate(fitz.open(PDF), start=1):
    captions = [(fitz.Rect(b[:4]), int(FIG.match(b[4])[1]))
                for b in page.get_text("blocks") if FIG.match(b[4]) and re.match(r"\s*Figure \d+\.", b[4])]
    for img in page.get_images(full=True):
        n += 1
        ref = reference_figure(page.get_image_rects(img[0])[0], captions)
        cap = by_index[n].get("caption") or ""
        m = FIG.match(cap)
        got = int(m[1]) if m else None
        if ref is None:
            verdict = "correct" if not cap else "spurious"
        elif got is None:
            verdict = "missing"
        elif got != ref:
            verdict = "wrong figure"
        elif not re.match(r"\s*Figure \d+\.", cap):
            verdict = "body text"
        else:
            verdict = "correct"
        rows.append({"img_doc_index": n, "page": page_num, "reference_figure": ref,
                     "extracted_figure": got, "verdict": verdict, "extracted_caption": cap[:200]})

df = pd.DataFrame(rows)
df.to_csv(HERE / "caption_check.csv", index=False)
print(df.verdict.value_counts().to_string())
print(f"{len(df)} images; extracted figure = image index whenever that figure is printed on the page: "
      f"{(df.extracted_figure == df.img_doc_index).sum()} cases")
