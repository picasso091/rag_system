from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

class DataIngestion:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
    
    def load(self) -> List[Document]:
        print(f"Loading PDF: {self.pdf_path}")
        docs = self.loader.load()
        print(f"Loaded {len(docs)} pages")
        return docs
