from dotenv import load_dotenv
from openai import OpenAI
import fitz 
import os
import pandas as pd
import time

load_dotenv(override=True, verbose=True)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY", "false")
)

PDF_PATH = "./experiments/gsgpy_70_5_3SO.pdf"
CSV_PATH = "./experiments/questions.csv"
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
                print(f"Testando {model} | Q{row['id']} | R{r}")
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
                    print(f"Erro no modelo {model}: {e}")
                    time.sleep(5)
                
    pd.DataFrame(results).to_csv("./experiments/concorrentes_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()