from langchain.llms import Ollama
from langchain.schema import Document
from typing import List

class LLMGeneration:
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434", temperature: float = 0.1):
        self.llm = Ollama(model=model, base_url=base_url, temperature=temperature)
    
    def generate(self, question: str, context_docs: List[Document]) -> str:
        context = self._format_context(context_docs)
        
        prompt = f"""You are analyzing Apple Inc.'s Q3 2022 financial filing.
        Answer ONLY based on the provided context. If you cannot find the answer, say so.
        Always cite specific numbers from the context.

        CONTEXT:
        {context}

        QUESTION: {question}

        ANSWER:"""
        
        print("Generating answer...")
        answer = self.llm.invoke(prompt)
        return answer
    
    def _format_context(self, docs: List[Document]) -> str:
        context_parts = []
        for doc in docs:
            context_parts.append(f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}")
        return "\n---\n".join(context_parts)
