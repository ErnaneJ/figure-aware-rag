"""Score generations with the same judge configuration as experiments/study1/evaluate.py
(DeepEval G-Eval Correctness, GPT-4.1 through OpenRouter). The evaluation steps are generated
once, saved, and reused so every condition is scored against the same rubric."""
import argparse, json, os, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from rag import HERE

load_dotenv(HERE.parent.parent / ".env", override=True)
os.environ["OPENAI_API_KEY"] = os.environ["OPEN_ROUTER_API_KEY"]
from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

JUDGE = GPTModel(model="openai/gpt-4.1", base_url="https://openrouter.ai/api/v1", timeout=90)
CRITERIA = ("Determine if the answer is factually correct and complete in relation to the Ground Truth. "
            "Rate it from 1 to 5.")
STEPS_FILE = HERE / "geval_steps.json"
PARAMS = [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]

def steps():
    if STEPS_FILE.exists():
        return json.loads(STEPS_FILE.read_text())
    m = GEval(name="Correctness", model=JUDGE, criteria=CRITERIA, evaluation_params=PARAMS)
    tc = LLMTestCase(input="x", actual_output="x", expected_output="x")
    m.measure(tc)
    STEPS_FILE.write_text(json.dumps(m.evaluation_steps, indent=1))
    return m.evaluation_steps

def score(row, gold, ev_steps):
    m = GEval(name="Correctness", model=JUDGE, evaluation_steps=ev_steps, evaluation_params=PARAMS, async_mode=False)
    tc = LLMTestCase(input=str(gold.question), actual_output=str(row.actual_output), expected_output=str(gold.ground_truth))
    try:
        m.measure(tc)
        return m.score, m.reason
    except Exception as e:
        return None, f"error: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(HERE / "generations.csv"))
    ap.add_argument("--out", default=str(HERE / "scores.csv"))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    gold = pd.read_csv(HERE.parent / "questions.csv").set_index("id")
    df = pd.read_csv(a.inp)
    df = df[df.actual_output.notna()]
    key = ["condition", "model", "id", "iteration"]
    out = Path(a.out)
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[prev.score_correctness.notna()]
        df = df.merge(prev[key], on=key, how="left", indicator=True)
        df = df[df._merge == "left_only"].drop(columns="_merge")
    if a.limit:
        df = df.head(a.limit)
    ev_steps = steps()
    print("to score", len(df), flush=True)
    cols = key + ["category", "score_correctness", "reason_correctness"]
    lock = threading.Lock()
    def work(r):
        s, why = score(r, gold.loc[r.id], ev_steps)
        with lock:
            rec = pd.DataFrame([[r.condition, r.model, r.id, r.iteration, r.category, s, why]], columns=cols)
            rec.to_csv(out, mode="a", header=not out.exists(), index=False)
            print(r.condition, r.model, r.id, r.iteration, s, flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, [r for _, r in df.iterrows()]))

if __name__ == "__main__":
    main()
