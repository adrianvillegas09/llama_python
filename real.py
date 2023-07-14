from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# loader = TextLoader("./1.txt")
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

with open("./1.txt") as f:
    contents = f.read()
texts = []
texts.append(contents)

db = FAISS.from_texts(
    texts,
    embeddings,
)

llm = GPT4All(
    model="./model.bin",
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    verbose=False,
)

# while True:
# command = input("Input : ")
res = qa(
    f"""
  What is B's answer in one word chunk for A's question?. If you can't find the correct answer, return 'None'.
"""
)
print(res["result"])
