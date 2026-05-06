from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("Input_and_Output/ca.pdf")
docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator=""
    )


# text="this is a langchain tutorial. you are going to learn a lot in this series"
# result=splitter.split_text(text)
result=splitter.split_documents(docs)

print(result[1].page_content)
print(result[1].metadata)