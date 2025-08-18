from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv(override=True, verbose=True)

def build_prompt(question: str, text_documents: list[Document], image_documents: list[Document]) -> str:
    filenames = sorted(set(d.metadata.get("filename") for d in text_documents + image_documents if d.metadata.get("filename")))

    available_files = "\n".join(f"- `{name}`" for name in filenames)

    text_context = "\n\n".join([
        f"[Arquivo: {d.metadata.get('filename')}] [Página {d.metadata.get('page_number')}] {d.page_content}"
        for d in text_documents
    ])

    image_context = "\n\n".join([
        f"[Imagem ID: {d.metadata['image_id']}] (Página {d.metadata.get('page_number')}) no arquivo `{d.metadata.get('filename')}`:"+
        f"\n- OCR: {d.metadata.get('ocr_text', '-- OCR não realizado --')}"
        f"\n- Descrição IA: {d.metadata.get('image_description', '-- descrição por IA não realizada')}"
        f"\n- Legenda: {d.metadata.get('image_caption', '-- legenda não fornecida no arquivo --')}"
        for d in image_documents
    ])
    
    prompt = (
        "Você é um assistente que responde com base no conteúdo textual e nas descrições de imagens extraídas de documentos PDF.\n\n"
        "### ARQUIVOS DISPONÍVEIS ###\n"
        f"{available_files}\n\n"
        "### TEXTOS EXTRAÍDOS ###\n"
        f"{text_context if text_context else '[nenhum texto disponível]'}\n\n"
        "### IMAGENS RELACIONADAS ###\n"
        f"{image_context if image_context else '[nenhuma imagem disponível]'}\n\n"
        "### INSTRUÇÕES ###\n"
        "- Quando for mencionar uma imagem, diga primeiro algo como: 'A Figura 7 (`página X`) do arquivo `nome.pdf` apresenta...'\n"
        "- Na **linha seguinte**, insira exatamente a tag correspondente à imagem: {{IMAGE:<id>}} e depois continue com a explicação.\n"
        "- Nunca coloque a tag no meio de frases. Sempre quebre a linha antes e depois dela.\n"
        "- Use apenas a descrição fornecida. Não invente conteúdos que não estão nela.\n"
        "- Ao final, inclua a fonte no formato: ***Fonte:** `nome_do_arquivo.pdf` — `página X`*\n\n"
        "- Nunca diga que não tem informações suficientes. Apenas use o que foi fornecido.\n"
        f"### PERGUNTA DO USUÁRIO ###\n{question}\n\n"
        "Responda com clareza, tom explicativo, e respeite as instruções de formatação."
    )
    
    # with open("debug_prompt.txt", "w") as f:
    #     f.write(prompt)
    return prompt