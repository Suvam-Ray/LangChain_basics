from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
model=ChatGroq(model="openai/gpt-oss-120b")

response=model.invoke("Hello, Langchain! Explain yourself in one sentence.")
#print(response)

# The above gave lot of information on tokens etc -- hence using below to just get the text
print(response.content)