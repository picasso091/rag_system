from src.pipeline import RAGPipeline
from config.settings import Config
import os

class InteractiveCLI:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
    
    def run(self):
        print("\n" + "="*60)
        print("Apple Q3 2022 - RAG System")
        print("="*60)
        print("Commands: 'q' to quit, 'clear' to clear screen\n")
        
        while True:
            try:
                question = input("❓ Ask: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['q', 'quit', 'exit']:
                    print("Goodbye!\n")
                    break
                
                if question.lower() == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                    continue
                
                result = self.pipeline.query(question)
                
                print("\n" + "="*60)
                print(f"Answer:")
                print("="*60)
                print(result["answer"])
                print("\n" + "-"*60)
                print("Sources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"{i}. Page {source['page']+1}: {source['content'][:100]}...")
                print("="*60 + "\n")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break
            except Exception as e:
                print(f"Error: {e}\n")
