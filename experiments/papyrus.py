"""Augmented condition of the 2026-02 experiment: runs every Golden Dataset question through the
Papyrus AI chat pipeline (inside the app container) and stores the answer and retrieved context.

Run inside the container (docker compose exec app python experiments/papyrus.py). Provider, model
and API keys come from the environment (.env, never committed; see .env.example)."""
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

from db.queries import create_chat, delete_chat
from services.chat_engine import run_chat

CSV_PATH = "./experiments/questions.csv"
FILE_HASH = "500f9ba505f61d63e7a587a20cb7bf0632a467e346b9ea916b854a649afe6f8f"
MODEL_NAME = os.getenv("LLM_MODEL_CHAT", "openai/gpt-4o")
OUTPUT = os.getenv("PAPYRUS_RESULTS_CSV", "results_papyrus_multimodel2.csv")


def test_papyrus():
    df = pd.read_csv(CSV_PATH)
    results = []

    for _, row in df.iterrows():
        for i in range(1, 4):
            chat_id = create_chat(name=f"Eval_Papyrus_{MODEL_NAME.split('/')[-1]}_Q{row['id']}", file_hash=FILE_HASH)
            print(f"Running: Papyrus + {MODEL_NAME} | Question: {row['id']} | Round: {i}")

            output = run_chat(chat_id, row['question'])

            context_chunks = [doc.page_content for doc in output.get("docs", [])]
            results.append({
                "id": row['id'],
                "category": row['category'],
                "model": f"Papyrus + {MODEL_NAME}",
                "iteration": i,
                "actual_output": output["message"].content,
                "retrieval_context": "\n---\n".join(context_chunks),
            })

            delete_chat(chat_id)

    pd.DataFrame(results).to_csv(OUTPUT, index=False)
    print(f"Done. File {OUTPUT} written.")


if __name__ == "__main__":
    test_papyrus()
