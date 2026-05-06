from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model1=ChatGroq(model_name="llama-3.3-70b-versatile")

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative']=Field(description="The sentiment of the feedback, must be either 'positive' or 'negative'")

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="classify the sentiment of the following text into positive or negative {feedback},\n{format_instruction}",
    input_variables=['feedback'],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

classifier_chain=prompt1 | model1 | parser2

prompt2=PromptTemplate(
    template="""write a warm appreciative and personalized response to this positive feedback. 
    Do not ask for any further input this should be a closing note. {classification_output}""",
    input_variables=['classification_output']
)

prompt3=PromptTemplate(
    template="""write a thoughful empathetic resolution-oriented response to this negative feedback. 
    Do not ask for any further input this should be a closing note. {classification_output}""",
    input_variables=['classification_output']
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=="positive", prompt2 | model1 | parser),
    (lambda x:x.sentiment == "negative", prompt3| model1 | parser),
    RunnableLambda(lambda x: "the sentiment is neutral ")
)

chain=classifier_chain | branch_chain

result=chain.invoke({"feedback":"My stay here was horrible. you people are stupid"})
print(result)