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
    model="./ggml-gpt4all-j-v1.3-groovy.bin",
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    verbose=False,
)

res = qa(
    f"""
    I want you to answer B's answer in a word.
    Extract it from the text.
"""
)
print(res["result"])
