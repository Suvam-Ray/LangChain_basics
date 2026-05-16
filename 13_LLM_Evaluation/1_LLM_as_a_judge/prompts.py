from langchain_core.prompts import PromptTemplate

QUESTION_PROMPT=PromptTemplate(
    template="Answer the following question as accurately as possible: \n {question}",
    input_variables=['question']
)

JUDGE_PROMPT=PromptTemplate(
    template="""
        You are an expert AI judge. Evaluate the following answer to the question \n\n
        question: {question}\n
        answer: {answer}\n\n
        Assess the answer based on the following criteria: \n
        1. accuracy (0-10)\n
        2. hallucination (true/false)\n
        3. feedback (brief comment) \n\n
        Return your evaluation as JSON only.
""",
    input_variables=["question", "answer"]
)
