from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "local_models/sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  print(f"📥 Baixando o modelo: {MODEL_NAME}")

  model = SentenceTransformer(MODEL_NAME)
  model.save(OUTPUT_DIR)

  print(f"✅ Modelo salvo em: {OUTPUT_DIR}")