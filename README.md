# RAG System V2 - Simple & Clean

A production-ready RAG system using LangChain, Chroma DB, and Mistral.

## Directory Structure

```
rag_system/
├── data/                     # Put your PDF here
│   └── 2022_Q3_AAPL.pdf
├── chroma_db/                # Chroma vector database
├── config/
│   └── settings.py           # Configuration
├── src/
│   ├── ingestion.py          # Data loading (PDF -> Documents)
│   ├── chunking.py           # Document chunking
│   ├── retrieval.py          # Vector search (Chroma DB)
│   ├── generation.py         # LLM answer generation
│   ├── pipeline.py           # RAG orchestrator
│   └── cli.py                # Interactive interface
├── main.py                   # Entry point
├── requirements.txt          # Dependencies
└── README.md
```

## RAG Pipeline Stages

1. **Data Ingestion** (`src/ingestion.py`)
   - Loads PDF documents
   - Converts to LangChain Documents

2. **Chunking** (`src/chunking.py`)
   - Splits documents into chunks
   - Preserves context with overlap

3. **Retrieval** (`src/retrieval.py`)
   - Stores chunks in Chroma DB
   - Performs semantic search
   - Uses HuggingFace embeddings

4. **Generation** (`src/generation.py`)
   - Uses Mistral LLM (via Ollama)
   - Generates answers from context

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama
```bash
ollama serve
```

### 3. Download Mistral Model
```bash
ollama pull mistral
```

### 4. Place PDF
Put your PDF in `data/` folder:
```
data/2022_Q3_AAPL.pdf
```

### 5. Run
```bash
python main.py
```

## Usage

### Interactive Mode
```bash
python main.py
```
Then ask questions:
```
❓ Ask: What were Apple's total net sales in Q3 2022?
```

### Programmatic Usage
```python
from src.pipeline import RAGPipeline
from config.settings import Config

pipeline = RAGPipeline(Config())
pipeline.build()  # First time only

result = pipeline.query("What was operating income?")
print(result["answer"])
```

## Configuration

Edit `config/settings.py`:
```python
CHUNK_SIZE = 1024           # Chunk size in characters
CHUNK_OVERLAP = 200         # Overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Embedding model
LLM_MODEL = "mistral"       # LLM model
TOP_K_RETRIEVAL = 5         # Documents to retrieve
```

## Vector Database

Chroma DB files are stored in `chroma_db/`:
- Auto-created on first run
- Persisted on disk
- Automatically loaded on subsequent runs

To reset: Delete `chroma_db/` folder

