# import all necessary libraries
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from utils import groq_llm, huggingface_instruct_embedding

def load_census_documents():
    loader = PyPDFDirectoryLoader('us-census-data')
    docs = loader.load()
    return docs
