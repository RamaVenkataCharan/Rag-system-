# import all necessary libraries
import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import ChatPromptTemplate
from utils import groq_llm, huggingface_instruct_embedding

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def create_rag_chain():
    document_chain = create_stuff_documents_chain(groq_llm(), prompt)
    retriever = st.session_state.vectors.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

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
