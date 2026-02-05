from src.pipeline import RAGPipeline
from config.settings import Config
import os

class InteractiveCLI:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
    
    def run(self):
        self._print_header()
        
        while True:
            try:
                question = input("\n>>> Ask: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['q', 'quit', 'exit']:
                    print("\nGoodbye!\n")
                    break
                
                if question.lower() == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                    self._print_header()
                    continue

                
                # Process query
                result = self.pipeline.query(question)
                self._display_result(result)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
    
    def _print_header(self):
        print("\n" + "="*70)
        print("Apple Q3 2022 - RAG System")
        print("="*70)
    
    def _display_result(self, result: dict):
        print("\n" + "="*70)
        print("ANSWER")
        print("="*70)
        print(result["answer"])
        
        print("\n" + "="*70)
        print("SOURCES")
        print("="*70)
        
        for i, source in enumerate(result["sources"], 1):
            page = source['page']
            page_display = page + 1 if isinstance(page, int) else page
            doc_type = source['type']
            content = source['content']
            
            if doc_type == "table":
                type_label = "TABLE"
                accuracy = source.get('accuracy', 'N/A')
                print(f"\n{i}.{type_label} (Page {page_display})")
            elif doc_type == "image":
                type_label = "IMAGE"
                print(f"\n{i}. {type_label} (Page {page_display})")
            else:
                type_label = "TEXT"
                print(f"\n{i}.{type_label} (Page {page_display})")
            
            preview = content if len(content) <= 150 else content[:147] + "..."
            print(f"   {preview}")
        

def main():
    from src.pipeline import RAGPipeline
    from config.settings import Config
    import sys
    
    config = Config()
    pipeline = RAGPipeline(config)
    
    # Check if we should build or load
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        pipeline.build()
    else:
        try:
            pipeline.load()
        except Exception as e:
            print(f"\n  Could not load existing pipeline: {e}")
            print("Building new pipeline...\n")
            pipeline.build()
    
    cli = InteractiveCLI(pipeline)
    cli.run()


if __name__ == "__main__":
    main()