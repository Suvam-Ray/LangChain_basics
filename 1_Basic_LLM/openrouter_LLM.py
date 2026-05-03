import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="minimax/minimax-m2.5:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

response = model.invoke("Hello, Langchain! Explain yourself in one sentence.")
# print(response)

# The above gave lot of information on tokens etc -- hence using below to just get the text
print(response.content)
