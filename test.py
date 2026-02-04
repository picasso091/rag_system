from src.pipeline import RAGPipeline
from config.settings import Config

pipeline = RAGPipeline(Config())
pipeline.build()  # First time only

result = pipeline.query("What was operating income?")
print(result["answer"])