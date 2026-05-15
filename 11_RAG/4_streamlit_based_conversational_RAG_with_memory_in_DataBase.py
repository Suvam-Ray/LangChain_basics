import streamlit as st
import sqlite3
import uuid
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    HumanMessage, 
    AIMessage,
    SystemMessage
)
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

load_dotenv()

# ---------------- DATABASE SETUP ----------------

# Creating sqlite connection for storing persistent chat history
# check_same_thread=False allows Streamlit to access DB from different threads
conn=sqlite3.connect("Input_and_Output/chat_memory1.db", check_same_thread=False)
cursor=conn.cursor()

# Creating chat_history table if it does not already exist
# Each row stores -> session_id, message role, and message content
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_history(
    session_id TEXT,
    role TEXT,
    content TEXT
    )
    """
)

conn.commit()

# Saving a single message into sqlite database
def save_message(session_id:str, role:str, content:str):
    cursor.execute(
        "INSERT INTO chat_history VALUES (?,?,?)",
        (session_id, role, content)
    )
    conn.commit()

# Loading full chat history for a given session_id from database
def load_chat_history(session_id:str)-> List[BaseMessage]:
    cursor.execute(
        "Select role, content from chat_history where session_id=?",(session_id,)
    )

    rows=cursor.fetchall()

    # Converting DB rows into LangChain message objects
    history: List[BaseMessage]=[]

    for role, content in rows:
        if role=="human":
            history.append(HumanMessage(content=content))
        elif role=="ai":
            history.append(AIMessage(content=content))
    return history

# Getting all unique previous session ids
def get_all_sessions():
    cursor.execute(
        "SELECT DISTINCT session_id from chat_history order by rowid desc"
    )
    return [row[0] for row in cursor.fetchall()]

# ---------------- STREAMLIT UI ----------------

# Streamlit page configurations
st.set_page_config(page_title="Conversational RAG", layout="wide")

# Main page title
st.title("Conversational RAG with memory")

# Sidebar
st.sidebar.title("Chats")

## Session management
# Generating a new session id for first app load
# Also creating empty in-memory chat history
if "session_id" not in st.session_state:
    st.session_state.session_id=str(uuid.uuid4())
    st.session_state.chat_history=[]

# Creating a "New Chat" button and starting a completely new conversation when button is clicked
if st.sidebar.button("New Chat"):
    st.session_state.session_id=str(uuid.uuid4())
    st.session_state.chat_history=[]

# Previous conversation separator
st.sidebar.markdown("Previous conversations")

# Showing all previous chat sessions in sidebar
# get_all_sessions() returns all unique session_ids stored in DB
for sid in get_all_sessions():

    # Creating one sidebar button per session
    # sid[:8] shows only first 8 characters to keep button text short
    # When user clicks a button -> condition becomes True
    if st.sidebar.button(sid[:8]):

        # Making clicked session the current active session
        st.session_state.session_id=sid

        # Loading full chat history of clicked session from database
        # and storing it in Streamlit session memory
        st.session_state.chat_history=load_chat_history(sid)

# Current active session id
session_id=st.session_state.session_id

# ---------------- PDF LOADING + VECTORSTORE ----------------

# Caching vectorstore so PDF embedding/indexing does not rerun every refresh
@st.cache_resource
def load_vectorstore():

    # Loading PDF document
    loader=PyPDFLoader("Input_and_Output/ca.pdf")
    document=loader.load()

    # Splitting PDF into smaller chunks for embedding/retrieval
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks=splitter.split_documents(document)

    # Embedding model used to convert chunks into vectors
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-miniLM-L6-v2"
    )

    # Creating Chroma vector database from document chunks
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore

# Loading cached vectorstore
vectorstore=load_vectorstore()

# Retriever used to fetch top 4 most relevant chunks
retriever=vectorstore.as_retriever(search_kwargs={"k":4})

# LLM initialization
llm=ChatGroq(model="llama-3.3-70b-versatile")

### In below ... understand prompt as = [list of 3 elements]
# element 1 : SystemMessage() -> contains system instructions for the LLM
# element 2 : MessagesPlaceholder("chat_history") -> placeholder where full conversation history will be inserted later during prompt.invoke()
# element 3 : Human message -> contains retrieved context and latest user question
prompt=ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful AI Assistant. Answer strictly from the provided context."
                "If the answer is not present, just say you don't know"
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Context: {context}, question: {input}"
        )
    ]
)

# Main conversational RAG function
def conversation_rag(user_input:str, chat_history: List[BaseMessage]):

    # Retrieving top relevant chunks from vector DB
    docs=retriever.invoke(user_input)

    # Combining retrieved chunks into one context string
    context="\n\n".join(
        f"[Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
        for d in docs
    )

    # Filling prompt template placeholders:
    # {input} -> latest user question
    # {context} -> retrieved document chunks
    # MessagesPlaceholder(chat_history) -> full previous conversation
    messages=prompt.invoke(
        {
            "input":user_input,
            "context":context,
            "chat_history":chat_history
        }
    )

    # Sending final messages list to LLM
    response=llm.invoke(messages)

    return response, docs

# Loading previous chat history for current session from DB
# Only loads if current in-memory history is empty
if not st.session_state.chat_history:
    st.session_state.chat_history=load_chat_history(session_id)

# ---------------- CHAT WINDOW ----------------

# Rendering/displaying all previous chat messages on screen
# st.session_state.chat_history contains full current conversation
# since we are using st.chat_message() components - streamlit understands that this is for a 
#     chat app and renders the right pane as a conversation
for msg in st.session_state.chat_history:

    # Checking if current message object is a HumanMessage
    # If yes -> display it as user chat bubble
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

    # Checking if current message object is an AIMessage
    # If yes -> display it as AI chat bubble
    elif isinstance(msg, AIMessage):
        st.chat_message("AI").write(msg.content)

# Creating chat input box at bottom of Streamlit app
# Whatever user types gets stored in user_input
user_input=st.chat_input("Ask a question from a PDF")

# Runs only when user enters something in chat input box
if user_input:

    # Immediately displaying latest user message in chat UI
    st.chat_message("user").write(user_input)

    # Persisting/storing user message in sqlite database
    # so conversation can be reloaded later
    save_message(session_id, "human", user_input)

    # Adding current user message into in-memory chat history
    # This history is later passed to MessagesPlaceholder(chat_history)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Running full conversational RAG pipeline:
    # 1. retrieve relevant chunks
    # 2. build prompt with context + chat_history
    # 3. send messages to LLM
    response,sources =conversation_rag(
        user_input, st.session_state.chat_history
    )

    # Displaying AI response in chat UI
    st.chat_message("AI").write(response.content)

    # Persisting/storing AI response in sqlite database
    save_message(session_id, "ai", response.content)

    # Adding AI response into current in-memory chat history
    # so future questions have conversation memory
    st.session_state.chat_history.append(AIMessage(content=response.content))
