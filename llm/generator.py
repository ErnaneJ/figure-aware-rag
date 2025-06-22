import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

def build_prompt(question: str, documents: list[Document]) -> str:
    context = "\n\n".join([
        f"[Arquivo: {d.metadata.get('filename')}] [Página {d.metadata.get('page_number')}] {d.page_content}"
        for d in documents
    ])

    prompt = (
        f"Você é um assistente que responde apenas com base nos documentos fornecidos.\n\n"
        f"CONTEÚDO EXTRAÍDO DOS ARQUIVOS:\n{context}\n\n"
        f"PERGUNTA DO USUÁRIO:\n{question}\n\n"
        f"Inclua, ao final da resposta, uma única linha como esta: "
        f"\n\n***Fonte:** `nome_do_arquivo.pdf` — `página X`*, "
        f"informando de onde você tirou a resposta. "
        f"Se a informação não estiver presente, diga claramente isso e não invente.\n\n"
        f"Responda de forma direta e factual."
    )
    return prompt


def generate_answer(question: str, documents: list[Document]) -> str:
    prompt = build_prompt(question, documents)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # ou gemini-2.0-flash se estiver disponível
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    response = llm.invoke(prompt)
    return response.content.strip()