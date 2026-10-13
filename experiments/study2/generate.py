"""Generation runs for the unified-settings and ablation study.
Every condition uses the same prompt template (llm/generator.py) and temperature 0.3;
conditions differ only in the context placed in the prompt."""
import argparse, csv, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rag import Index, records, text_chunks, image_chunks, build_prompt, FILENAME, HERE

load_dotenv(HERE.parent.parent / ".env", override=True)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPEN_ROUTER_API_KEY"], timeout=90, max_retries=1)
TEMPERATURE, MAX_TOKENS = 0.3, 1200
MODELS = ["meta-llama/llama-3.1-70b-instruct", "deepseek/deepseek-chat", "google/gemini-2.5-flash-lite"]
PROVIDER = {"meta-llama/llama-3.1-70b-instruct": {"order": ["DeepInfra"], "allow_fallbacks": True}}

R = records()
CONDITIONS = {
    "FT": None,  # full document text in the TEXTS block, no retrieval
    "PAGE_TEXT": (("page", {}), None),
    "PAGE_FULL": (("page", {}), ("caption", "ocr", "desc")),  # original Papyrus configuration
    "SUB_TEXT": (("subpage", dict(size=800, overlap=200)), None),
    "SUB_OCRCAP": (("subpage", dict(size=800, overlap=200)), ("caption", "ocr")),
    "SUB_FULL": (("subpage", dict(size=800, overlap=200)), ("caption", "ocr", "desc")),
}

def context_for(cond):
    if CONDITIONS[cond] is None:
        docs = [{"text": p["text"], "page": p["page"], "source": "text"} for p in R["pages"] if p["text"]]
        return lambda q: docs
    (mode, kw), parts = CONDITIONS[cond]
    idx = Index(text_chunks(R["pages"], mode, **kw) + (image_chunks(R["images"], parts) if parts else []))
    return idx.retrieve

def call(model, prompt):
    for attempt in range(4):
        try:
            extra = {"usage": {"include": True}}
            if model in PROVIDER:
                extra["provider"] = PROVIDER[model]
            r = client.chat.completions.create(model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                               messages=[{"role": "user", "content": prompt}], extra_body=extra)
            u = r.usage
            cost = getattr(u, "cost", None)
            if cost is None and getattr(u, "model_extra", None):
                cost = u.model_extra.get("cost")
            return r.choices[0].message.content, u.prompt_tokens, u.completion_tokens, cost, getattr(r, "provider", None)
        except Exception as e:
            print("retry", model, e)
            time.sleep(5 * (attempt + 1))
    return None, None, None, None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--questions", nargs="*", type=int)
    ap.add_argument("--out", default=str(HERE / "generations.csv"))
    ap.add_argument("--records", default="records.json", help="records.json, or records_layout.json for the layout-caption rerun")
    a = ap.parse_args()
    global R
    R = records(a.records)
    q = pd.read_csv(HERE.parent / "questions.csv")
    if a.questions:
        q = q[q.id.isin(a.questions)]
    out = Path(a.out)
    done = set()
    if out.exists():
        d = pd.read_csv(out)
        d = d[d.actual_output.notna()]
        done = set(zip(d.condition, d.model, d.id, d.iteration))
    fields = ["condition", "model", "id", "category", "iteration", "actual_output", "retrieval_context",
              "n_chunks", "prompt_tokens", "completion_tokens", "cost", "provider"]
    new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()
    jobs = []
    for cond in a.conditions:
        retrieve = context_for(cond)
        for _, r in q.iterrows():
            docs = retrieve(r.question)
            prompt = build_prompt(r.question, docs)
            ctx = "\n---\n".join(d["text"] for d in docs)
            for m in a.models:
                for it in range(1, a.reps + 1):
                    if (cond, m, r.id, it) not in done:
                        jobs.append((cond, m, r, it, prompt, ctx, len(docs)))
    print("jobs", len(jobs))
    def run(job):
        cond, m, r, it, prompt, ctx, n = job
        t0 = time.time()
        ans, pt, ct, cost, prov = call(m, prompt)
        print(f"{cond} {m} Q{r.id} R{it} {time.time() - t0:.1f}s", flush=True)
        return {"condition": cond, "model": m, "id": r.id, "category": r.category, "iteration": it,
                "actual_output": ans, "retrieval_context": ctx if cond != "FT" else "", "n_chunks": n,
                "prompt_tokens": pt, "completion_tokens": ct, "cost": cost, "provider": prov}
    total = 0.0
    with ThreadPoolExecutor(4) as ex:
        for row in ex.map(run, jobs):
            w.writerow(row); fh.flush()
            total += row["cost"] or 0
    print(f"done, cost this run ${total:.4f}")

if __name__ == "__main__":
    main()
