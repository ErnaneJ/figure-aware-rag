"""Locate the evidence for each Golden Dataset question in the PDF text (page and character offset)
by best token-overlap window against the ground-truth answer."""
import re, json
import pandas as pd
from rag import records, HERE

R = records()
q = pd.read_csv(HERE.parent / "questions.csv")
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
rows = []
for _, r in q.iterrows():
    gt = set(tok(r.ground_truth)) - {"the", "a", "of", "and", "to", "in", "is", "by", "into", "that", "this"}
    best = (0, None, None)
    for p in R["pages"]:
        words = [(m.group(0).lower(), m.start()) for m in re.finditer(r"[A-Za-z0-9]+", p["text"])]
        W = max(len(gt) * 2, 12)
        for i in range(0, max(1, len(words) - W), 3):
            win = {w for w, _ in words[i:i + W]}
            s = len(gt & win) / max(1, len(gt))
            if s > best[0]:
                best = (s, p["page"], words[i][1])
    rows.append({"id": r.id, "category": r.category, "ref": r.source_reference, "pdf_page": best[1],
                 "offset": best[2], "overlap": round(best[0], 2),
                 "page_len": len(R["pages"][best[1] - 1]["text"]) if best[1] else None})
df = pd.DataFrame(rows)
df["beyond_1800"] = df.offset > 1800
df.to_csv(HERE / "evidence_map.csv", index=False)
print(df.to_string())
