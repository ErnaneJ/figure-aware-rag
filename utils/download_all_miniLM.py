import os
from sentence_transformers import SentenceTransformer

def download_model():
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    OUTPUT_DIR = "local_models/sentence-transformers/all-MiniLM-L6-v2"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📥 Baixando o modelo: {MODEL_NAME}")
    
    if not os.path.exists(os.path.join(OUTPUT_DIR, "config.json")):
        model = SentenceTransformer(MODEL_NAME)
        model.save(OUTPUT_DIR)
        print(f"✅ Modelo salvo em: {OUTPUT_DIR}")
    else:
        print("✅ Modelo já existe, pulando download.")

if __name__ == "__main__":
    download_model()