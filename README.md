# End-to-End RAG System using ObjectBox and LangChain

An on-device, high-performance Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ObjectBox Vector Database**, **Groq (LLaMA3-8B)**, and **Streamlit**.

This application allows users to ask questions and retrieve accurate, context-grounded insights from US Census demographic PDF reports directly on-device with sub-second retrieval and LLM response times.

![Streamlit Web App Interface](./images/RAG%20app%20UI.png)

---

## Key Features

- **On-Device Vector Storage**: Embedded with [ObjectBox](https://objectbox.io/), a high-performance vector database that stores and queries embeddings locally without external database latency.
- **Ultra-Fast LLM Inference**: Powered by Groq Cloud running `Llama3-8b-8192` for lightning-fast token generation.
- **Dense Semantic Embeddings**: Utilizes Hugging Face's `BAAI/bge-small-en-v1.5` embeddings (768 dimensions) for semantic retrieval.
- **Interactive UI**: Built with Streamlit, offering query input, response latency tracking, and transparent similarity search inspection with context chunks.
- **PDF Document Ingestion**: Ingests, parses, and chunks complex PDF reports using `PyPDFDirectoryLoader` and `RecursiveCharacterTextSplitter`.

---

## Architecture & Workflow

```
[ US Census PDFs ]
       │
       ▼
[ PyPDFDirectoryLoader ] ──► [ RecursiveCharacterTextSplitter ]
                                            │
                                            ▼
                           [ BAAI/bge-small-en-v1.5 ]
                                            │
                                            ▼
                                [ ObjectBox Vector DB ]
                                            │
User Query ──────────────────────► [ Retriever ]
                                            │
                                            ▼
                                  [ Retrieved Chunks ]
                                            │
                                            ▼
                                [ Groq LLaMA3-8b-8192 ]
                                            │
                                            ▼
                               [ Streamlit UI Output ]
```

---

## Tech Stack & Libraries

- **Language**: Python 3.10+
- **Framework**: LangChain (`langchain`, `langchain-community`, `langchain-core`, `langchain-groq`, `langchain-objectbox`)
- **Vector Database**: ObjectBox Vector Store
- **LLM**: Groq (`Llama3-8b-8192`)
- **Embeddings**: HuggingFace BGE (`BAAI/bge-small-en-v1.5`)
- **Document Parser**: PyPDF
- **Frontend**: Streamlit

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git
- Groq API Key (get free access at [Groq Console](https://console.groq.com/keys))

### 2. Clone the Repository
```bash
git clone https://github.com/RamaVenkataCharan/Rag-system-.git
cd Rag-system-
```

### 3. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Key
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Add your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 6. Run the Application
```bash
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Embed Documents**: Click the **"Embed Documents"** button to load the US Census PDFs, split them into chunks of 1000 characters, compute embeddings, and store them in ObjectBox. (Pre-embedded vectors are also included in `objectbox/`).
2. **Ask Questions**: Type any question related to the demographic census data (e.g., *"What is the poverty rate by age group?"* or *"What are the computer ownership trends?"*).
3. **Inspect Context**: Open the **"Document Similarity Search"** expander to inspect the exact source chunks retrieved from the documents.

---

## Project Structure

```
Rag-system-/
├── .gitignore
├── .env.example
├── LICENSE
├── README.md
├── requirements.txt
├── app/
│   ├── app.py             # Streamlit web application & RAG chain
│   ├── config.py          # Environment configuration loader
│   └── utils.py           # LLM and embedding factory functions
├── images/
│   └── RAG app UI.png     # UI application screenshot
├── objectbox/             # Persistent local ObjectBox vector database
│   ├── data.mdb
│   └── lock.mdb
└── us-census-data/        # PDF source census reports
    ├── acsbr-015.pdf
    ├── acsbr-016.pdf
    ├── acsbr-017.pdf
    └── p70-178.pdf
```

---

## Author

- **Rama Venkata Charan**
- GitHub: [@RamaVenkataCharan](https://github.com/RamaVenkataCharan)
- Repository: [Rag-system-](https://github.com/RamaVenkataCharan/Rag-system-)

---

## Acknowledgments

- [ObjectBox Database](https://objectbox.io/)
- [LangChain Framework](https://www.langchain.com/)
- [Groq AI](https://groq.com/)
- [Krish Naik](https://www.youtube.com/@krishnaik06)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
