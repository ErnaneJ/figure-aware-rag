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

    return f"""You must answer exclusively based on the texts and descriptions below.
        FILES:
        {available_files}

        TEXTS:
        {text_context if text_context else "[none]"}

        IMAGES:
        {image_context if image_context else "[none]"}

        RULES:
        - Use English to answer, even if the question is in another language.
        - For images, write: "Figure X (page Y) of file Z shows:"
        - On the next line, insert exactly:
        {{IMAGE:<id>}}
        - Do not invent information.
        - Always reference the source of the information with: 'Source: file — page'. If there is more than one, list them all. The same applies to images.

        QUESTION:
        {question}
        """
