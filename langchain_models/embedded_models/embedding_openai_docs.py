from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", dimensions=32)


documents = [
    "Washington DC is the capital of the United States.",
    "The capital of France is Paris.",  
    "The capital of Japan is Tokyo."
]



result = embeddings.embed_documents(documents)

print(str(result))