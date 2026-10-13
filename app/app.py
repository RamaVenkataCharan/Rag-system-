import os
import sys
import time
import streamlit as st

# Ensure current app directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import ChatPromptTemplate
from utils import groq_llm, huggingface_instruct_embedding

# Base paths for census data and objectbox
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'us-census-data')
DB_DIR = os.path.join(BASE_DIR, 'objectbox')

st.set_page_config(layout='wide', page_title="Objectbox and Langchain RAG")

st.title('Objectbox VectorstoreDB with LLAMA3')

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Auto-load existing ObjectBox vectorstore if present on disk
if 'vectors' not in st.session_state:
    if os.path.exists(DB_DIR) and os.path.exists(os.path.join(DB_DIR, 'data.mdb')):
        try:
            embeddings = huggingface_instruct_embedding()
            st.session_state.embeddings = embeddings
            st.session_state.vectors = ObjectBox(
                embedding=embeddings,
                embedding_dimensions=768,
                db_directory=DB_DIR
            )
        except Exception:
            pass

def vector_embedding():
    with st.spinner("Processing documents and generating embeddings..."):
        st.session_state.embeddings = huggingface_instruct_embedding()
        st.session_state.loader = PyPDFDirectoryLoader(DATA_DIR)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:200])
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensions=768,
            db_directory=DB_DIR
        )

col1, col2 = st.columns([1, 4])
with col1:
    if st.button('Embed Documents'):
        vector_embedding()
        st.success('ObjectBox Database is ready. You can now enter your question.')

if 'vectors' in st.session_state:
    st.info('ObjectBox Vector Store is loaded and ready.')
else:
    st.warning('ObjectBox database is not loaded yet. Click "Embed Documents" above to initialize.')

user_input = st.text_input('Enter your question from documents:')

if user_input:
    if 'vectors' not in st.session_state:
        st.error("Please click 'Embed Documents' first to initialize the vector database!")
    else:
        try:
            document_chain = create_stuff_documents_chain(groq_llm(), prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()

            response = retrieval_chain.invoke({'input': user_input})
            elapsed = time.process_time() - start

            st.write(response['answer'])
            st.caption(f'Response time: {elapsed:.2f} seconds')

            with st.expander("Document Similarity Search (Context Chunks)"):
                for i, doc in enumerate(response.get("context", [])):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.divider()
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
