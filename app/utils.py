from langchain_groq import ChatGroq
from config import load_config, get_groq_api

# load app configuration
load_config()

# setup groq LLM
def groq_llm():
    llm = ChatGroq(groq_api_key=get_groq_api(), model_name='Llama3-8b-8192')
    return llm
