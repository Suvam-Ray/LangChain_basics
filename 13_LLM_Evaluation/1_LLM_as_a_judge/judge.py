from langchain_groq import ChatGroq
from prompts import JUDGE_PROMPT
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class Evaluation(BaseModel):
    accuracy: int =Field(description="Score from 0 to 10 where 10 is highest or best score")
    hallucination: bool =Field(description="true if the answer contains hallucination else false")
    feedback: str =Field(description="Brief comment on the quality of the answer")

def evaluate_answer(question: str, answer: str)-> dict:
    judge_llm=ChatGroq(model_name="llama-3.3-70b-versatile")
    structured_judge_llm=judge_llm.with_structured_output(Evaluation)
    prompt=JUDGE_PROMPT.format(question=question, answer=answer)
    response=structured_judge_llm.invoke(prompt)
    return response


if __name__=="__main__":
    question="What is the capital of France?"
    answer="It is Berlin"
    result=evaluate_answer(question, answer)
    print("Evaluation result: ", result)
