"""Check that the offline replica reproduces the retrieval contexts stored in study1/answers_papyrus.csv."""
import pandas as pd
from rag import Index, records, text_chunks, image_chunks, HERE

R = records()
idx = Index(text_chunks(R["pages"], "page") + image_chunks(R["images"]))
q = pd.read_csv(HERE.parent / "questions.csv").set_index("id")
res = pd.read_csv(HERE.parent / "study1" / "answers_papyrus.csv")
exact = setmatch = 0
for qid, grp in res.groupby("id"):
    stored = {c.strip() for c in str(grp.retrieval_context.iloc[0]).split("\n---\n")}
    same_all = grp.retrieval_context.nunique() == 1
    got = {d["text"].strip() for d in idx.retrieve(q.loc[qid, "question"])}
    ok = stored == got
    setmatch += ok
    if not ok:
        print(qid, "stored", len(stored), "got", len(got), "overlap", len(stored & got), "stable_across_runs", same_all)
print(f"questions with identical retrieved set: {setmatch}/{q.shape[0]}")
