from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", dimensions=32)


result = embeddings.embed_query("Hi, how are you?")

print(str(result))