import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def extract_caption_with_llm(
    page_text: str,
    filename: str,
    page_number: int,
    img_doc_index: int
) -> str | None:
    """
    Usa LLM para identificar e retornar a legenda da imagem `img_doc_index` com base no texto da página.
    Retorna apenas a legenda exata, ou None se não houver.
    """

    question = (
        f"A imagem {img_doc_index} nesta página possui alguma legenda visível no texto?"
        f" Se sim, retorne apenas a legenda exata como ela aparece no texto, sem interpretar ou reescrever."
        f" A legenda pode estar em qualquer parte do texto da página, mas geralmente está próxima da imagem."
        f" O conteúdo pode iniciar com palavras como 'Figura', 'Imagem', 'Foto', 'Gráfico', 'Tabela' ou similares,"
        f" e pode conter números, letras ou símbolos. A legenda pode ser longa ou curta,"
        f" e pode conter pontuação, como dois pontos, travessão ou parênteses."
        f" Caso não exista nenhuma legenda claramente relacionada a essa imagem, diga apenas: 'UNDEFINED'."
    )

    context = (
        f"Você é um assistente que deve responder com base no conteúdo textual da página.\n\n"
        f"ARQUIVO: {filename} | PÁGINA: {page_number} | IMAGEM: {img_doc_index}\n\n"
        f"CONTEÚDO DA PÁGINA:\n{page_text}\n\n"
        f"PERGUNTA:\n{question}\n\n"
        f"Responda exclusivamente com a legenda exata ou com o termo UNDEFINED."
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )

    response = llm.invoke(context)
    answer = response.content.strip()

    if "UNDEFINED" in answer.upper():
        return None

    return answer
