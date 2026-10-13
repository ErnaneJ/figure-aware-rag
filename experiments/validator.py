from dotenv import load_dotenv
from openai import OpenAI
import fitz 
import os
import pandas as pd
import time

load_dotenv(override=True, verbose=True)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY", "false")
)

PDF_PATH = "./experiments/gsgpy_70_5_3SO.pdf"
CSV_PATH = "./experiments/questions.csv"
MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemini-2.5-flash-lite-preview-09-2025"
]

if __name__ == "__main__":
  doc = fitz.open(PDF_PATH)
  pdf_context = ""
  for page_num, page in enumerate(doc, start=1):
      text = page.get_text()
      pdf_context += f"Page {page_num}: {text.strip()}\n"
  
  print(len(pdf_context))
  
  # client = OpenAI(
  #   base_url="https://openrouter.ai/api/v1",
  #   api_key=os.getenv("OPEN_ROUTER_API_KEY", "false")
  # )
  
  # for model in MODELS:
  #   completion = client.chat.completions.create(
  #     model=model,
  #     messages=[
  #         {"role": "user", "content": f"Hi! How are you?!"}
  #     ],
  #     temperature=0.4
  #   )
    
  #   print(f"Model: {model} | Response: {completion.choices[0].message.content}")