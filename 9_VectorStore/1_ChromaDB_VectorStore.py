from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()

texts=[
    "Large language models are trained on massive datasets",
    "Large language models(llms) are particularly trained using transformers",
    "chroma is a lightweight vector storeused in langchain",
    "embeddings convert text into numerical represetnation"
]

embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma.from_texts(
    texts=texts,
    embedding=embedding,
    collection_name="langchain_chroma_demo"
)

query="tell me more about llms"
results=vectorstore.similarity_search(query, k=2)

print(results)