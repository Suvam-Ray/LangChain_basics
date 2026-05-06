from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("Input_and_Output/ca.pdf")
docs=loader.load()

print(docs[0].metadata)
print(docs[0])