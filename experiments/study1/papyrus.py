"""Study 1, Papyrus condition: runs every benchmark question through the Papyrus AI chat pipeline
and stores the answer and the retrieved context (20 questions x 3 runs per model).

Run inside the app container after uploading the PDF in the UI, once per chat model:
    LLM_MODEL_CHAT=openai/gpt-4o docker compose exec app python experiments/study1/papyrus.py
The study used TEXT_CHUNKING=page. Answers from the five runs were concatenated into answers_papyrus.csv.
FILE_HASH is the SHA-256 of the PDF, which names its folder under files/."""
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

from db.queries import create_chat, delete_chat
from services.chat_engine import run_chat

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent / "questions.csv"
FILE_HASH = "500f9ba505f61d63e7a587a20cb7bf0632a467e346b9ea916b854a649afe6f8f"
MODEL_NAME = os.getenv("LLM_MODEL_CHAT", "openai/gpt-4o")
OUTPUT = os.getenv("PAPYRUS_RESULTS_CSV", str(HERE / f"answers_papyrus_{MODEL_NAME.split('/')[-1]}.csv"))


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
