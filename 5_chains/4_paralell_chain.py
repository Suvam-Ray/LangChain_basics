from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGroq(model_name="llama-3.3-70b-versatile")

model2 = ChatGroq(model_name="llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
    template="Gnerate short and simple note from following topic {text_input}",
    input_variables=['text_input']
)
prompt2=PromptTemplate(
    template="generate 5 shoor question answer from the following text {text_input}",
    input_variables=['text_input']
)

prompt3=PromptTemplate(
    template="Merge the provided notes and question answers into single document {note_generated}, {quiz_generated}",
    input_variables=['note_generated', 'quiz_generated']
)

parser=StrOutputParser()

runnable_chain=RunnableParallel(
    {
        'note_generated':prompt1 | model1 | parser,
        'quiz_generated':prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

final_chain = runnable_chain | merge_chain

text="""
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.\n
The advantages of support vector machines are:\n
- Effective in high dimensional spaces.\n
- Still effective in cases where number of dimensions is greater than the number of samples.\n
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.\n
- Versatile: different kernel functions can be specified for the decision function. Common kernels are provided, but it is also \n
possible to specify custom kernels.
\n\n
The disadvantages of support vector machines include:\n
- If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions\n
and regularization term is crucial.\n
- SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold \n
cross-validation (see Scores and probabilities, below).\n
- The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) \n
and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, \n
it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix \n
(sparse) with dtype=float64.\n
"""

result = final_chain.invoke(text)
print(result)