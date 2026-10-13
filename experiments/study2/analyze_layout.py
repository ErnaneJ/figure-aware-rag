"""Rerun of the caption-dependent conditions with captions matched by page layout.

The original ingestion gave only 6 of the 32 figures their own caption (study1/caption_check.py).
rebuild_records.py --captions layout writes records_layout.json, with the same OCR text and VLM
descriptions and the captions from indexing/captions.py. generate.py and judge.py then rerun
PAGE_FULL, SUB_OCRCAP and SUB_FULL into generations_layout.csv and scores_layout.csv, with the
same models, prompt, temperature, repetitions and judge. FT, PAGE_TEXT and SUB_TEXT use no
captions and keep their original runs.

Reads both runs; writes tables/layout_rerun.json and prints a summary. No API calls.
Same statistics as analyze.py: question means (N=10 per category), paired two-sided Wilcoxon,
r = |Z|/sqrt(N), 95% bootstrap CIs over questions. Bonferroni k=7 over the same family."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import FAMILY, GOLD_IMAGE, wilcoxon
from rag import Index, records, text_chunks, image_chunks

HERE = Path(__file__).parent
KEY = ["condition", "model", "id", "iteration"]
RERUN = ["PAGE_FULL", "SUB_OCRCAP", "SUB_FULL"]
FIG1, FIG10_12 = [11, 17, 18, 19, 20], [12, 13, 14, 15, 16]
CHUNKING = {"PAGE_FULL": ("page", {}, ("caption", "ocr", "desc")),
            "SUB_OCRCAP": ("subpage", dict(size=800, overlap=200), ("caption", "ocr")),
            "SUB_FULL": ("subpage", dict(size=800, overlap=200), ("caption", "ocr", "desc"))}


def scored(gen, sc):
    sc = pd.read_csv(HERE / sc).drop_duplicates(KEY, keep="last")
    sc = sc[sc.score_correctness.notna()]
    return pd.read_csv(HERE / gen).merge(sc[KEY + ["score_correctness"]], on=KEY, how="inner")


def changed_questions():
    """Questions whose retrieved context differs between the two caption versions."""
    q = pd.read_csv(HERE.parent / "questions.csv")
    old, new = records(), records("records_layout.json")
    out = {}
    for c, (mode, kw, parts) in CHUNKING.items():
        a = Index(text_chunks(old["pages"], mode, **kw) + image_chunks(old["images"], parts))
        b = Index(text_chunks(new["pages"], mode, **kw) + image_chunks(new["images"], parts))
        out[c] = [int(r.id) for _, r in q.iterrows()
                  if [d["text"] for d in a.retrieve(r.question)] != [d["text"] for d in b.retrieve(r.question)]]
    return out


def main():
    old = scored("generations.csv", "scores.csv")
    new = scored("generations_layout.csv", "scores_layout.csv")
    res = {"n_new_scored": len(new), "n_new_generations": len(pd.read_csv(HERE / "generations_layout.csv")),
           "cost_new_generations": float(pd.read_csv(HERE / "generations_layout.csv").cost.sum())}
    mixed = pd.concat([old[~old.condition.isin(RERUN)], new])
    qo = old.groupby(["condition", "id", "category"]).score_correctness.mean().reset_index()
    qn = mixed.groupby(["condition", "id", "category"]).score_correctness.mean().reset_index()

    def qmean(q, c, ids=None, cat=None):
        s = q[q.condition == c]
        if cat:
            s = s[s.category == cat]
        if ids:
            s = s[s.id.isin(ids)]
        return s.set_index("id").score_correctness.sort_index()

    res["pooled"] = {c: {cat: {"original": round(qmean(qo, c, cat=cat).mean(), 3),
                               "layout": round(qmean(qn, c, cat=cat).mean(), 3)}
                         for cat in ("text", "visual")} for c in RERUN}
    res["visual_groups"] = {c: {g: {"original": round(qmean(qo, c, ids).mean(), 3),
                                    "layout": round(qmean(qn, c, ids).mean(), 3)}
                                for g, ids in (("figure_1", FIG1), ("figures_10_12", FIG10_12))}
                            for c in ["SUB_TEXT"] + RERUN}

    # Pre-declared family, recomputed with the rerun conditions
    tests = []
    for name, a, b, cat in FAMILY:
        A, B = qmean(qn, a, cat=cat), qmean(qn, b, cat=cat)
        t = wilcoxon(A.values, B.reindex(A.index).values)
        t.update(comparison=name, category=cat, a=a, b=b, mean_a=A.mean(), mean_b=B.mean(),
                 p_bonf=min(1.0, t["p"] * len(FAMILY)))
        tests.append({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in t.items()})
    extra = [("OCR+caption over text, sub-page", "SUB_OCRCAP", "SUB_TEXT", "visual")]
    for name, a, b, cat in extra:
        A, B = qmean(qn, a, cat=cat), qmean(qn, b, cat=cat)
        t = wilcoxon(A.values, B.reindex(A.index).values)
        t.update(comparison=name, category=cat, a=a, b=b, mean_a=A.mean(), mean_b=B.mean(), p_bonf=None)
        tests.append({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in t.items()})
    res["tests"] = tests

    # Same condition, original vs layout captions
    res["layout_vs_original"] = []
    for c in RERUN:
        for cat in ("text", "visual"):
            A, B = qmean(qn, c, cat=cat), qmean(qo, c, cat=cat)
            t = wilcoxon(A.values, B.reindex(A.index).values)
            res["layout_vs_original"].append({"condition": c, "category": cat,
                                              **{k: round(float(v), 4) for k, v in t.items()}})

    # Run-to-run variation: questions whose context did not change get the same prompt in both runs
    ch = changed_questions()
    res["changed_context_questions"] = ch
    drift = []
    for c in RERUN:
        same = [i for i in range(1, 21) if i not in ch[c]]
        d = (qmean(qn, c, same) - qmean(qo, c, same)).abs()
        drift.append({"condition": c, "n_questions": len(same), "mean_abs_diff": round(d.mean(), 3),
                      "mean_diff": round((qmean(qn, c, same) - qmean(qo, c, same)).mean(), 3)})
    res["rerun_drift_unchanged_context"] = drift

    # Is the image record of the figure a visual question asks about retrieved at all?
    q = pd.read_csv(HERE.parent / "questions.csv").set_index("id")
    gold = {}
    for name in ("records.json", "records_layout.json"):
        R = records(name)
        for c, (mode, kw, parts) in CHUNKING.items():
            idx = Index(text_chunks(R["pages"], mode, **kw) + image_chunks(R["images"], parts))
            gold[f"{c}|{name}"] = {i: [d["image_id"] for d in idx.retrieve(q.loc[i, "question"]) if d["source"] == "image"]
                                   for i in GOLD_IMAGE}
    res["retrieved_images"] = gold
    res["gold_image_retrieved"] = {k: sum(GOLD_IMAGE[i] in v[i] for i in GOLD_IMAGE) for k, v in gold.items()}

    (HERE / "tables").mkdir(exist_ok=True)
    (HERE / "tables" / "layout_rerun.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
