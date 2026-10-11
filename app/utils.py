import os
import sys

# Ensure app directory is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import load_config, get_groq_api

# load app configuration
load_config()

# setup groq LLM
def groq_llm():
    api_key = get_groq_api()
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please set GROQ_API_KEY in your .env file.")
    llm = ChatGroq(groq_api_key=api_key, model_name='Llama3-8b-8192')
    return llm

# setup huggingface_instruct_embedding
def huggingface_instruct_embedding():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name='BAAI/bge-small-en-v1.5',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings
