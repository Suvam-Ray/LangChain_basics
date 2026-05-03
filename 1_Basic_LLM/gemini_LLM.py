from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

response=llm.invoke("Hello, Langchain! Explain yourself in one sentence.")

#print(response)

# The above gave lot of information on tokens etc -- hence using below to just get the text
print(response.content)
