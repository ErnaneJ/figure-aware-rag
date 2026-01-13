from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv(override=True, verbose=True)

def build_prompt(question: str, text_documents, image_documents) -> str:
    filenames = sorted({
        d.metadata.get("filename")
        for d in text_documents + image_documents
        if d.metadata.get("filename")
    })

    available_files = "\n".join(f"- {name}" for name in filenames)

    text_context = "\n\n".join(
        f"({d.metadata.get('filename')} – pág. {d.metadata.get('page_number')})\n{d.page_content}"
        for d in text_documents
    )

    image_context = "\n\n".join(
        f"(Imagem {d.metadata['image_id']} – pág. {d.metadata.get('page_number')} – {d.metadata.get('filename')})\n"
        f"{d.page_content}"
        for d in image_documents
    )

    return f"""Você responde exclusivamente com base nos textos e descrições abaixo.
        ARQUIVOS:
        {available_files}

        TEXTOS:
        {text_context if text_context else "[nenhum]"}

        IMAGENS:
        {image_context if image_context else "[nenhuma]"}

        REGRAS:
        - Para imagens, escreva:
        "A Figura X (pág. Y) do arquivo Z mostra:"
        - Na linha seguinte, insira exatamente:
        {{IMAGE:<id>}}
        - Não invente informações.
        - Sempre referencie a fonte da informação com: 'Fonte: arquivo — página'. Se mais de uma, liste todas. O mesmo vale para imagens.

        PERGUNTA:
        {question}
        """
