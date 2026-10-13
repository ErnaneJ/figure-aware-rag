# Papyrus AI

Retrieval-augmented generation (RAG) over PDF documents, with figures turned into searchable text at ingestion time.

Most PDF question-answering pipelines keep only the text layer, so a language model never sees what is inside figures. Papyrus AI describes each embedded image once, when the document is uploaded, with a vision-language model. It joins that description with the figure caption and the text that OCR reads in the image, and indexes the result next to the page text. At query time any text-only LLM receives the top retrieved chunks, and every answer cites file and page.

![Chat interface](./assets/preview.png)

## How it works

1. **Extraction.** PyMuPDF splits each page into a text track and an image track. Images are deduplicated by MD5.
2. **Visual enrichment (once per image).** Tesseract OCR reads printed text, a vision-language model (Gemini 2.5 Flash by default) writes a description, and an LLM extracts the figure caption from the page text. The three fields are joined into one image record.
3. **Text chunking.** Each page is cut into overlapping windows (800 characters, 200 overlap) that keep their page number. The original page-level chunking is still available (`TEXT_CHUNKING=page`).
4. **Indexing.** Text chunks and image records are embedded with `all-MiniLM-L6-v2` (384-d) into one FAISS index per document.
5. **Grounded answer.** A LangGraph workflow retrieves the top 5 chunks per document, builds a prompt with TEXTS and IMAGES blocks and grounding rules, and asks the chat model to answer only from that context and to cite `Source: <file>, page <n>`.

![Architecture](./assets/flow--arq.png)

| Component | Role |
|---|---|
| Streamlit + LangGraph (`app`) | Chat UI and retrieval/generation workflow |
| Celery (`worker`) + Redis | Asynchronous ingestion |
| PostgreSQL (`db`) | Chats, files, pages, image records, messages |
| FAISS (shared volume) | One vector index per document |
| LLM providers | Gemini, OpenAI-compatible APIs (OpenRouter, Groq, DeepSeek), Ollama for local models |

## Running

Requirements: Docker and Docker Compose.

```bash
cp .env.example .env                   # provider, model and API key for each role
python ./utils/download_all_miniLM.py  # embedding model used by the indexer
docker-compose build
docker-compose up                      # app on http://localhost:28502
```

Each LLM role is configured separately in `.env`: chat (`LLM_*_CHAT`), image description (`LLM_*_IMAGE_DESCRIBER`) and caption extraction (`LLM_*_CAPTIONS`). Useful switches:

| Variable | Default | Effect |
|---|---|---|
| `TEXT_CHUNKING` | `subpage` | `subpage` (overlapping windows) or `page` (first 1,800 characters per page) |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `800`, `200` | Window size for `subpage` |
| `DISABLE_OCR`, `DISABLE_MULTIMODAL_DESCRIPTION`, `DISABLE_LLM_CAPTION_EXTRACTION` | unset | Turn off parts of the image record |

To use local models, run [Ollama](https://ollama.com) and set `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

To reset the database: `docker-compose exec app python ./db/init_db.py`.

## Experiments

The `experiments/` folder holds the evaluation reported in the M.Sc. dissertation and the accompanying article. The benchmark has 20 questions (10 answered in the prose, 10 answered only in figures) on S. Chopra and K. J. Marfurt, "Seismic attributes: A historical perspective," *Geophysics*, 70(5), 2005, [doi:10.1190/1.2098670](https://doi.org/10.1190/1.2098670). The PDF is copyrighted and is not distributed here. To rerun the experiments, obtain it through the DOI and save it as `experiments/gsgpy_70_5_3SO.pdf`.

| Path | Content |
|---|---|
| `experiments/questions.csv` | Benchmark questions and reference answers |
| `experiments/papyrus.py`, `experiments/evaluate.py` | Study 1: full-text baseline vs. Papyrus on five models (600 answers), judged with G-Eval (GPT-4.1) |
| `experiments/AVALIACAO_FINAL_MESTRADO.csv` | Study 1 judged answers |
| `experiments/revision2/` | Study 2: six conditions under one prompt and temperature (720 answers): full text, page vs. sub-page chunks, with and without image records, OCR + caption vs. full record |

Study 2 pipeline (needs `OPEN_ROUTER_API_KEY` in `.env`):

```bash
cd experiments/revision2
python rebuild_records.py     # page text + image records from the PDF and the ingestion log (no API calls)
python validate_replica.py    # checks the offline retriever against the logged retrieval
python retrieval_study.py     # evidence recall at k=5 for each chunking (no API calls)
python generate.py            # answers for all conditions
python judge.py               # G-Eval correctness
python analyze.py             # tables/, figs/ and analysis.json
```

Main results on this benchmark: retrieval cuts prompt tokens by more than 93% and cost by 91% to 95% against sending the whole document. Sub-page chunking recovers most of the accuracy lost on text questions, and image records raise accuracy on figure questions, mostly through OCR text and captions. With 10 questions per category the differences are exploratory. See the dissertation and the article for the full analysis.

## Citation

E. F. Rocha Junior, "Ingestion-Time Visual Enrichment for Retrieval-Augmented Generation over Scientific Documents: Cost and Accuracy Trade-offs in a Geophysics Case Study," M.Sc. dissertation, PPgEEC, Universidade Federal do Rio Grande do Norte, Natal, Brazil, 2026.

E. F. Rocha Junior, "LLMs e Vetorização Semântica para Interação Multimodal com Documentos PDF: Uma Abordagem de Geração Aumentada por Recuperação (RAG)," undergraduate thesis, UFRN, 2025. [Repository](https://repositorio.ufrn.br/items/0d91f59a-21de-4477-b27f-236c469d7661).

## Acknowledgments

Supported by Petrobras through the "Cognitive Assistant for Geophysical Data" project, with support from ANP through the R&D levy regulation, by CAPES, and by FUNPEC.
