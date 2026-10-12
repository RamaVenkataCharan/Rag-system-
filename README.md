# End to End RAG Project using ObjectBox and LangChain

An on-device, high-performance Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ObjectBox Vector Database**, and **Groq LLM**.

![Streamlit Web App Interface](./images/RAG%20app%20UI.png)

## Description
This project showcases the implementation of an advanced RAG system that uses Objectbox vector database and Groq's LLaMA3 model as an LLM to retrieve information from US Census demographic PDF documents.

### Workflow:
1. Load PDF documents from `us-census-data` using `PyPDFDirectoryLoader`.
2. Chunk text using `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200).
3. Store vector embeddings generated via `HuggingFaceBgeEmbeddings` in `ObjectBox` vector store.
4. Setup `ChatGroq` with `Llama3-8b-8192`.
5. Assemble retrieval and document chains for responsive QA.
