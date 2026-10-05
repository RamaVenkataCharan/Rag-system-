# import all necessary libraries
import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from utils import groq_llm, huggingface_instruct_embedding

def load_and_split_documents():
    loader = PyPDFDirectoryLoader('us-census-data')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs[:200])
    return split_docs
