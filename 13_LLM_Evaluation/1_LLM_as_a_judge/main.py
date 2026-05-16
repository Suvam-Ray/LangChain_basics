from langchain_groq import ChatGroq
from prompts import QUESTION_PROMPT
from judge import evaluate_answer
from dotenv import load_dotenv

load_dotenv()

llm=ChatGroq(model_name="llama-3.3-70b-versatile")

def generate_answer(question: str)->str:
    prompt=QUESTION_PROMPT.format(question=question)
    
    response=llm.invoke(prompt)
    return response.content

def ask_until_good(question:str, threshold: int=8, max_attempts: int =3):
    attempts=0
    while attempts<max_attempts:
        answer=generate_answer(question)
        print("question:", question)
        print("answer: ", answer)
        evaluation=evaluate_answer(question, answer)
        print("Evaluation:", evaluation)

        if evaluation.accuracy>=threshold and not evaluation.hallucination:
            print("Answer meets the threshold, no correction needed")
            print("Answer")
            print('\n\n')
            break
        else:
            print("Answer is below threshold, regenerating ...")
            attempts+=1

if __name__=="__main__":
    questions=["What is the capital of France?",
               "Who wrote the book Harry Potter?",
               "Name the largest and most dense city in the world",
               "Is Israel trying to invade Palestine?"]

    for question in questions:
        ask_until_good(question)
#    answer=generate_answer(question)
#    print("question:", question)
#    print("answer: ", answer)
#    evaluation=evaluate_answer(question, answer)
#    print("Evaluation:", evaluation)
#    print("\n\n")
