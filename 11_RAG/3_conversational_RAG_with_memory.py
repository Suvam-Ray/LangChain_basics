from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

loader=PyPDFLoader("Input_and_Output/ca.pdf")
documents=loader.load()

llm=ChatGroq(model="llama-3.3-70b-versatile")

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks=splitter.split_documents(documents)

embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-miniLM-L6-v2")

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding
)

retriever=vectorstore.as_retriever(search_kwargs={"k":4})
### In below ... understand prompt as = [list of 3 elements]
# element 1 : SystemMessage() -> contains system instructions for the LLM
# element 2 : MessagesPlaceholder("chat_history") -> placeholder where full conversation history will be inserted later during prompt.invoke()
# element 3 : Human message -> contains retrieved context and latest user question
prompt=ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=("""

You are a helpful AI Assistant. Asnwer strictly from the provided context. If the answer is
                     not present in the context, just say you don't know""")
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Context: \n {context} \n\n question: \n{input}"
        )
    ]
)

def conversational_rag(user_nput:str, chat_history: List[BaseMessage]):
    docs=retriever.invoke(user_nput)
    context="\n\n".join(
        f"[Page {d.metadata.get('page','N/A')}]\n{d.page_content}" for d in docs
    )

    # Filling prompt template placeholders:
    # {input} -> latest user question
    # {context} -> retrieved document chunks
    # MessagesPlaceholder(chat_history) -> full previous conversation
    messages=prompt.invoke(
        {
            "input":user_nput,
            "context":context,
            "chat_history":chat_history                    
        }
    )

    response=llm.invoke(messages)

    return response, docs

chat_history: List[BaseMessage]=[]


print("Conversational RAG")

while True:
    user_input=input("you:")
    if user_input.lower()=="exit":
        break
    chat_history.append(HumanMessage(content=user_input))
    responses, sources= conversational_rag(user_input, chat_history=chat_history)

    chat_history.append(AIMessage(content=responses.content))
    print("\nAI:",responses.content)