from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model=ChatGroq(model_name="llama-3.3-70b-versatile")

class Person(BaseModel):
    name:str=Field(description="The person's full name")
    age:int=Field(gt=18, lt=60, description="The person's age, must not be less than 18 and greater than 60")
    city:str=Field(description="The city where the person lives in")

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template=("give me the name, age and city of a fictional {place} person\n"
              "Make sure the age is greater than 18.\n"
              "Return the response in the following format:\n\n"
              "{format_instruction}\n\n"),
    input_variables=["place"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

# prompt=template.invoke({'place':'Austria'})
chain=template | model | parser
result=chain.invoke({"place": "Austria"})
print(result)