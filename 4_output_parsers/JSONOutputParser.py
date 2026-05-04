from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model=ChatGroq(model_name="llama-3.3-70b-versatile")
parser=JsonOutputParser()

template=PromptTemplate(
    template="Give me the name, age and city of a fictional person. and the name and city has to be indian. {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

chain=template | model | parser
result=chain.invoke({})
print(result)