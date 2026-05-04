from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

print("blog post generator")
print("Provide ideas or topics for the blog post. Type exit to finish")

topic = input("Enter the blog post topic: ")

chat_prompt_template=ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a professional blog writer. Help generate informative, engaging, and well structured blog post about a {topic}"),
    HumanMessagePromptTemplate.from_template("Write a detailed blog post about {topic}")
])

# Initialize chat history
chat_history=[]

while True : 
    user_input = input("Please share ideas or instructions. You can type exit to quit the conversation :")

    if user_input.lower()=="exit":
        print("Exitting the blog post generator")
        break

    # Contructing the message trail -- in 3 steps
    # Step 1: Adding the first system prompt and user message using prompt template
    messages = chat_prompt_template.format_messages(topic=topic)

    # Step 2: Appending previous history to messages
    for prev in chat_history:
        messages.append(prev)

    # Step 3: Appending the latest user message provided
    current_user_message = HumanMessage(content=user_input)
    messages.append(current_user_message)

    # Invoking model to get the response
    response = chat_model.invoke(messages)
    print("\nBlog Post Content:\n", response.content)

    # Appending user message and AI response to chat history
    chat_history.append(current_user_message)
    chat_history.append(AIMessage(content=response.content))
