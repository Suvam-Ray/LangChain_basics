import sqlite3
import requests

# Download the Chinook SQLite database file
url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
response = requests.get(url)

# Create and populate the local database
connection = sqlite3.connect("Input_and_Output/Chinook.db")
connection.cursor().executescript(response.text)
connection.close()

print("Database 'Chinook.db' created successfully!")


import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# 1. Connect to the Database
db = SQLDatabase.from_uri("sqlite:///Input_and_Output/Chinook.db")

# 2. Setup the LLM (Keep temperature 0 for SQL to prevent syntax errors)
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 3. Initialize the Toolkit
# This gives the agent tools like 'sql_db_query', 'sql_db_schema', etc.
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 4. Create the Agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="openai-tools", # This is the most reliable for SQL tasks
)

# 5. Ask a complex data question
response = agent_executor.invoke({
    "input": "Who are the top 3 best-selling artists and how much did they earn?"
})

print(response["output"])