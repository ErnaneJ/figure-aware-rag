"""Study 1, baseline condition: each model receives the full text of the PDF (PyMuPDF) and the question.

5 models x 20 questions x 3 runs = 300 answers, temperature 0.4. Writes answers_baseline.csv.
Needs OPEN_ROUTER_API_KEY in .env and the source PDF at experiments/gsgpy_70_5_3SO.pdf."""
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import fitz
import os
import pandas as pd
import time

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent.parent / ".env", override=True)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY", "false")
)

PDF_PATH = HERE.parent / "gsgpy_70_5_3SO.pdf"
CSV_PATH = HERE.parent / "questions.csv"
MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemini-2.5-flash-lite-preview-09-2025"
]

def run_experiment():
    df_questions = pd.read_csv(CSV_PATH)
    
    doc = fitz.open(PDF_PATH)
    pdf_context = ""
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        pdf_context += f"Page {page_num}: {text.strip()}\n"
    
    results = []

    for model in MODELS:
        for _, row in df_questions.iterrows():
            for r in range(1, 4):
                print(f"Running {model} | Q{row['id']} | R{r}")
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a specialized assistant. Answer the question strictly using the provided context."},
                            {"role": "user", "content": f"Context: {pdf_context}\n\nQuestion: {row['question']}"}
                        ],
                        temperature=0.4
                    )
                    
                    answer = completion.choices[0].message.content
                    results.append({
                        "id": row['id'],
                        "category": row['category'],
                        "model": model,
                        "iteration": r,
                        "actual_output": answer
                    })
                except Exception as e:
                    print(f"Error with {model}: {e}")
                    time.sleep(5)
                
    pd.DataFrame(results).to_csv(HERE / "answers_baseline.csv", index=False)

if __name__ == "__main__":
    run_experiment()
