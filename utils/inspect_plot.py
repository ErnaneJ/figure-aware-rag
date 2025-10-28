from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, perplexity=5, random_state=42)
# reduced = tsne.fit_transform(vectors)

# === CONFIGURAÇÃO ===
file_hash = "fe523df48ad2837d60c3b3b2e227594a911f06c2b990a215e6ff8cc44de4fa4f"
index_path = Path("files") / file_hash / "vectorstore"

# === CARREGAR ===
embedding_model = HuggingFaceEmbeddings(model_name="local_models/sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(str(index_path), embedding_model, allow_dangerous_deserialization=True)

faiss_index = vectorstore.index
vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)

# === REDUÇÃO DE DIMENSÃO ===
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

# === OBTÉM METADADOS ===
docs = list(vectorstore.docstore._dict.values())
page_numbers = [doc.metadata.get("page_number") for doc in docs]

# === PLOTAR ===
plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c='blue')

for i, page in enumerate(page_numbers):
    plt.text(reduced[i, 0], reduced[i, 1], str(page), fontsize=9)

plt.title(f"Distribuição vetorial das páginas — {file_hash[:8]}...")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.show()
