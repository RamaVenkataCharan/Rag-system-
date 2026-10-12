# End to End RAG Project using ObjectBox and LangChain

An on-device, high-performance Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ObjectBox Vector Database**, **Groq (LLaMA3-8B)**, and **Streamlit**.

![Streamlit Web App Interface](./images/RAG%20app%20UI.png)

## Description
This project showcases the implementation of an advanced RAG system that uses ObjectBox vector database and Groq's LLaMA3 model as an LLM to retrieve information from US Census demographic PDF documents.

### Key Architecture:
1. Ingest PDF documents from `us-census-data` using `PyPDFDirectoryLoader`.
2. Chunk text with `RecursiveCharacterTextSplitter` (chunk size 1000, chunk overlap 200).
3. Create embeddings with `BAAI/bge-small-en-v1.5` and store them in `ObjectBox`.
4. Initialize `ChatGroq` (`Llama3-8b-8192`) for sub-second text generation.
5. Chain retriever and LLM prompt using `create_retrieval_chain`.

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/RamaVenkataCharan/Rag-system-.git
   cd Rag-system-
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY
   ```

5. **Run the Application:**
   ```bash
   streamlit run app/app.py
   ```
