from dotenv import load_dotenv
load_dotenv()

# 1.load a pdf document
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("Input_and_Output/ca.pdf")
documents=loader.load()

# 2. Split documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs=text_splitter.split_documents(documents)

# 3. create embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-miniLM-L6-v2"
)

# 4. store embeddings in vector database (chroma)
from langchain_chroma import Chroma

vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="rag_collection"
)

# 5. retriever 

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":2, "lambda_mult":0.5}
)

# 6. Initialize LLM
from langchain_groq import ChatGroq

llm=ChatGroq(model_name="llama-3.3-70b-versatile")

# 7. create prompttemplate
from langchain_core.prompts import PromptTemplate

prompt=PromptTemplate(
    template="""
You are an AI assistant. Use the following context to answer the question.
If the answer is not present in the context, say you don't know.

<context_here>
{context}
</context_here>

<question_here>
{question}
</question_here>
""",
input_variables=["context", "question"]
)

# 8. OutputParser definition

from langchain_core.output_parsers import StrOutputParser

parser=StrOutputParser()

# 9. Build a RAG chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# In below - the output from retriver is fed into format_docs function -- to concatenate. 
# and question is the query from the user. 
rag_chain=(
    {
        "context":retriever | format_docs,
        "question": lambda query: query
    } | prompt | llm | parser
)

# 10. ask a question
query="What is Architecture of a CPU"

response=rag_chain.invoke(query)
print(response)