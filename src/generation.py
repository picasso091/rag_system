from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from typing import List
import base64
import requests
import json

class LLMGeneration:
    def __init__(
        self, 
        model: str = "mistral", 
        base_url: str = "http://localhost:11434", 
        temperature: float = 0.1,
        vision_model: str = "llava",
        use_vision: bool = True
    ):
        self.llm = OllamaLLM(model=model, base_url=base_url, temperature=temperature)
        self.base_url = base_url
        self.vision_model = vision_model
        self.use_vision = use_vision
    
    def generate(self, question: str, context_docs: List[Document]) -> str:
        processed_docs = self._process_image_documents(context_docs)
        
        context = self._format_context(processed_docs)
        
        prompt = f"""You are analyzing Apple Inc.'s Q3 2022 financial filing.
        Answer ONLY based on the provided context. If you cannot find the answer, say so.
        Always cite specific numbers from the context.

        The context may include:
        - Text content from the document
        - Tables with financial data
        - Image descriptions from charts and figures

        CONTEXT:
        {context}

        QUESTION: {question}

        ANSWER:"""
        
        answer = self.llm.invoke(prompt)
        return answer
    
    def _process_image_documents(self, docs: List[Document]) -> List[Document]:
        processed_docs = []
        
        for doc in docs:
            if doc.metadata.get("type") == "image" and self.use_vision:
                if "[IMAGE_PLACEHOLDER" in doc.page_content:
                    try:
                        image_data = doc.metadata.get("image_data")
                        if image_data:
                            description = self._describe_image(image_data)
                            
                            new_doc = Document(
                                page_content=f"IMAGE DESCRIPTION: {description}",
                                metadata=doc.metadata.copy()
                            )
                            processed_docs.append(new_doc)
                        else:
                            processed_docs.append(doc)
                    except Exception as e:
                        print(f"Warning: Could not process image: {e}")
                        processed_docs.append(doc)
                else:
                    processed_docs.append(doc)
            else:
                processed_docs.append(doc)
        
        return processed_docs
    
    def _describe_image(self, image_base64: str) -> str:
        try:
            # Call Ollama API with vision model
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.vision_model,
                "prompt": "Describe this image in detail. Focus on any charts, graphs, tables, numbers, or financial information visible. Be specific about data points and trends.",
                "images": [image_base64],
                "stream": False
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Could not generate image description")
            else:
                return f"Error generating image description (status: {response.status_code})"
        
        except Exception as e:
            return f"Error generating image description: {str(e)}"
    
    def _format_context(self, docs: List[Document]) -> str:
        context_parts = []
        
        for doc in docs:
            doc_type = doc.metadata.get("type", "text")
            page_num = doc.metadata.get("page", "N/A")
            
            if doc_type == "table":
                context_parts.append(f"[TABLE from Page {page_num + 1}]\n{doc.page_content}\n")
            elif doc_type == "image":
                context_parts.append(f"[IMAGE from Page {page_num + 1}]\n{doc.page_content}\n")
            else:
                context_parts.append(f"[TEXT from Page {page_num + 1}]\n{doc.page_content}\n")
        
        return "\n---\n".join(context_parts)
    
    def generate_with_sources(self, question: str, context_docs: List[Document]) -> dict:
        answer = self.generate(question, context_docs)
        
        sources = []
        for doc in context_docs:
            source_info = {
                "page": doc.metadata.get("page", "N/A"),
                "type": doc.metadata.get("type", "text"),
                "content": doc.page_content[:500]
            }
            
            if doc.metadata.get("type") == "table":
                source_info["table_accuracy"] = doc.metadata.get("accuracy", "N/A")
            
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources
        }