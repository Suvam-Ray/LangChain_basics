from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatGroq(model_name="llama-3.3-70b-versatile")

prompt=PromptTemplate(
    template="answer the following question {question} from the following text {text}",
    input_variables=['question', 'text']
)

url="https://techlekh.com/dongfeng-nammi-vigo-price-nepal/"

loader=WebBaseLoader(url)

docs=loader.load()

parser=StrOutputParser()

chain=prompt | model | parser

result=chain.invoke({'question':"What is the product the article talking about?", 'text':docs[0].page_content})
print(result)