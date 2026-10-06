# import all necessary libraries
import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_objectbox.vectorstores import ObjectBox
from utils import groq_llm, huggingface_instruct_embedding

def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = huggingface_instruct_embedding()
        st.session_state.loader = PyPDFDirectoryLoader('us-census-data')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:200])
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensions=768,
            db_directory='objectbox'
        )
