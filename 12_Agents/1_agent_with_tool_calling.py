from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

llm=ChatGroq(
    model="llama-3.3-70b-versatile"
)

@tool
def multiply(a:int, b:int)-> int:
    """Multiply two numbers"""
    return a*b

@tool
def add(a:int, b:int)-> int:
    """Add two numbers"""
    return a+b

tools=[add, multiply]

system_prompt="""
You are a helpful AI Agent. Use tools when necessary. If no tool is required, answer directly
"""

agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

print("LangChain Basic Agent. Type exit to quit\n")

while True:
    user_input=input("You:")
    if user_input.lower()=="exit":
        break
    response=agent.invoke(
        {"messages":[{'role':'user','content': user_input}]}
    )
    print("\nAI:", response['messages'][-1].content)
    print("\n =========AI Agent Debug Mode==============\n", response)
    print("\n\n")