## Quickstart

Install deps and activate a venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Ollama and pull a model:

```bash
ollama serve
ollama pull mistral
```

Build and run:

```bash
python main.py --build
python main.py
```

To reset vectors, delete the `chroma_db/` folder.
