"""Retrieval-only study (no LLM calls): evidence recall at k=5 for each indexing configuration."""
import pandas as pd
from rag import Index, records, text_chunks, image_chunks, embedder, HERE

R = records()
q = pd.read_csv(HERE.parent / "questions.csv")
ev = pd.read_csv(HERE / "evidence_map.csv").set_index("id")
GOLD_IMAGE = {11: 2, 17: 2, 18: 2, 19: 2, 20: 2, 12: 11, 13: 12, 14: 12, 15: 13, 16: 13}
tokzr = embedder().tokenizer

def hit(qid, docs):
    if qid in GOLD_IMAGE:
        return any(d["source"] == "image" and d["image_id"] == GOLD_IMAGE[qid] for d in docs)
    e = ev.loc[qid]
    return any(d["source"] == "text" and d["page"] == e.pdf_page and d["start"] <= e.offset < d["end"] for d in docs)

CONFIGS = {
    "page1800": ("page", {}), "sub500": ("subpage", dict(size=500, overlap=100)),
    "sub800": ("subpage", dict(size=800, overlap=200)), "sub1000": ("subpage", dict(size=1000, overlap=200)),
}
IMG = {"none": None, "ocr+cap": ("caption", "ocr"), "full": ("caption", "ocr", "desc")}
rows = []
for cname, (mode, kw) in CONFIGS.items():
    tdocs = text_chunks(R["pages"], mode, **kw)
    ntok = [len(tokzr(d["text"])["input_ids"]) for d in tdocs]
    indexed_chars = sum(d["end"] - d["start"] for d in tdocs) if mode == "page" else sum(len(p["text"]) for p in R["pages"])
    for iname, parts in IMG.items():
        docs = tdocs + (image_chunks(R["images"], parts) if parts else [])
        idx = Index(docs)
        hits = {r.id: hit(r.id, idx.retrieve(r.question)) for _, r in q.iterrows()}
        rows.append({"chunks": cname, "images": iname, "n_text_chunks": len(tdocs),
                     "pct_over_256_tokens": round(100 * sum(t > 256 for t in ntok) / len(ntok), 1),
                     "text_chars_indexed": indexed_chars,
                     "recall_text": sum(hits[i] for i in range(1, 11)) / 10,
                     "recall_visual": sum(hits[i] for i in range(11, 21)) / 10,
                     "hits": "".join("1" if hits[i] else "0" for i in range(1, 21))})
df = pd.DataFrame(rows)
df.to_csv(HERE / "retrieval_study.csv", index=False)
print(df.to_string())
