# 1. pip install langchain-ollama
# 2. prepare tools for llm
from langchain_ollama import ChatOllama


def build_llm():
    llm = ChatOllama(base_url="https://ollama.cwh0628.qzz.io/api/generate", model="gemma4", temperature=0.2)
    return llm

