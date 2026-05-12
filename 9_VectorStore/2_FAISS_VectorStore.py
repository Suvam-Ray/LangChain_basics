from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

cricket_texts=[
    "Cricket is one of the most popular sports in the world, especially in countries like India, England, Australia, and Nepal.",
    "Sachin Tendulkar is known as the God of Cricket because of his consistency, technique, and long international career.",
    "Cricket is played in three major formats: Test cricket, One Day Internationals, and Twenty20.",
    "Each cricket team consists of eleven players including batsmen, bowlers, all-rounders, and a wicket-keeper.",
    "The International Cricket Council (ICC) governs cricket worldwide and organizes tournaments like the Cricket World Cup.",
    "Cricket requires physical fitness, strategy, teamwork, and strong mental discipline."
]

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=FAISS.from_texts(
    texts=cricket_texts,
    embedding=embeddings
)

query="Who is the God of cricket?"
results=vectorstore.similarity_search(query,k=1)
print(results)