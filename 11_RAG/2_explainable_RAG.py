from dotenv import load_dotenv

load_dotenv()


# 1. Load PDF
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("Input_and_Output/ca.pdf")
documents=loader.load()

# 2. Split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks=text_splitter.split_documents(documents)

# 3. Create embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store embeddings in Chroma
from langchain_chroma import Chroma

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding= embeddings,
    collection_name="source-aware-rag"
)

# 5. similarity search

query="What is the report about?"

retrieved_docs=vectorstore.similarity_search(
    query=query,
    k=3
)

# build context+source
context_text=""
sources=[]

for i, doc in enumerate(retrieved_docs):
    page=doc.metadata.get('page',"N/A")
    source=doc.metadata.get("source", "PDF")

    context_text += f"\n Chunk {i+1}: \n{doc.page_content}\n"

    sources.append({
        "chunk":i+1,
        "page": page,
        "source":source,
        "content":doc.page_content
    })

# 7. prompt llm with context
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm=ChatGroq(
    model="llama-3.3-70b-versatile"
)

prompt=PromptTemplate(
    template="""
    Answe the question only using the context below. Also provide the actual chunk during retrieval
    along with the page number of the document where i can find that chunk

    Context: {context}

    Question: {question}
    """,
    input_variables=['context','question']
)

parser=StrOutputParser()

chain= prompt | llm | parser

answer=chain.invoke({
    "context":context_text,
    "question":query
})

print("################# LLM Response start ######################")
print(answer)
print("################# LLM Response end ######################")
print("\n\n################# Printing chunks now ######################")

for src in sources:
    print("-----------------------------------------")
    print(f"Chunk: {src['chunk']} | Page : {src['page']}")
    print(src['content'])