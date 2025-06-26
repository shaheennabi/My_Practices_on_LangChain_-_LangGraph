from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=64)

docs = [
    "LangChain makes it easier to build LLM-powered applications.",
    "FAISS is a library for efficient similarity search of embeddings.",
    "Transformers are the foundation of most modern language models.",
    "OpenAI provides powerful APIs for text generation and embeddings.",
    "Vector databases are used to store and retrieve embeddings."
]


user_query = "What is LangChain?"

# Generate embeddings for the documents and the user query
doc_embeddings = embeddings.embed_documents(docs)
user_query_embedding = embeddings.embed_query(user_query)

# Calculate cosine similarity between the user query embedding and document embeddings
similarities = cosine_similarity([user_query_embedding], doc_embeddings)[0]

index, score = print("Cosine Similarities:", sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1])

print(f"Most similar document index: {index}, Score: {score}")
