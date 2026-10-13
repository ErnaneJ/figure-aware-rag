"""Rebuild the ingestion records (page text, OCR, caption, VLM description) of the
benchmark document from the PDF and the original ingestion log, without new API calls."""
import argparse, hashlib, json, re, sys
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "gsgpy_70_5_3SO.pdf"
LOG = HERE.parent / "study1" / "ingestion_log.txt"

HEAD = re.compile(r"^worker-1  \| \[(?P<ts>[^\]]+): (?P<lvl>\w+)/[^\]]+\] (?P<msg>.*)$")

def parse_log():
    records, cur = [], None
    for line in LOG.read_text(encoding="utf-8").splitlines():
        m = HEAD.match(line)
        if m:
            cur = {"ts": m["ts"], "msg": m["msg"]}
            records.append(cur)
        elif cur is not None:
            cur["msg"] += "\n" + re.sub(r"^worker-1  \| ", "", line)
    images, current = {}, None
    for r in records:
        msg = r["msg"]
        m = re.match(r"Extraindo OCR para .*/extracted/(img(\d+)-page(\d+)_img_(\d+)\.\w+)", msg)
        if m:
            current = m[1]
            images[current] = {"file": m[1], "img_doc_index": int(m[2]), "page": int(m[3]),
                               "img_page_index": int(m[4]), "t_start": r["ts"]}
            continue
        if current is None:
            continue
        for key, prefix in (("ocr_text", "OCR extraído: "), ("llm_description", "Descrição obtida: "),
                            ("caption", "Legenda obtida: ")):
            if msg.startswith(prefix):
                val = msg[len(prefix):].strip()
                images[current][key] = None if val == "None" else val
                if key == "caption":
                    images[current]["t_end"] = r["ts"]
    return images

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", choices=["log", "layout"], default="log",
                    help="log: captions as extracted at ingestion (records.json). layout: captions matched by "
                         "page layout with indexing/captions.py (records_layout.json); OCR and VLM fields unchanged.")
    a = ap.parse_args()
    if a.captions == "layout":
        sys.path.insert(0, str(HERE.parent.parent))
        from indexing.captions import caption_blocks, image_rect, layout_caption
    log_images = parse_log()
    doc = fitz.open(PDF)
    pages, images, seen = [], [], {}
    idx = 1
    for pno, page in enumerate(doc, start=1):
        pages.append({"page": pno, "text": page.get_text().strip()})
        for ipi, img in enumerate(page.get_images(full=True), start=1):
            data = doc.extract_image(img[0])
            h = hashlib.md5(data["image"]).hexdigest()
            name = f"img{idx}-page{pno}_img_{ipi}.{data['ext']}"
            if h in seen:
                src = seen[h]
                rec = {**src, "file": name, "img_doc_index": idx, "page": pno, "img_page_index": ipi, "is_duplicate": True}
            else:
                rec = dict(log_images[name])
                rec["is_duplicate"] = False
                if a.captions == "layout":
                    rect = image_rect(page, img[0])
                    rec["caption"] = layout_caption(page, rect, caption_blocks(page)) if rect is not None else None
                seen[h] = rec
            images.append(rec)
            idx += 1
    out = {"sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(), "pages": pages, "images": images}
    (HERE / ("records.json" if a.captions == "log" else "records_layout.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    uniq = [i for i in images if not i["is_duplicate"]]
    print("sha256", out["sha256"][:16], "pages", len(pages), "images", len(images), "unique", len(uniq),
          "log entries", len(log_images))
    print("with caption", sum(bool(i.get("caption")) for i in uniq), "with ocr", sum(bool(i.get("ocr_text")) for i in uniq),
          "with desc", sum(bool(i.get("llm_description")) for i in uniq))
    print("total chars", sum(len(p["text"]) for p in pages))

if __name__ == "__main__":
    main()
