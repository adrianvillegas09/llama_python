from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(
    ["A: What is your zip code? B: It's 90232. not 23212"],
    embeddings,
    persist_directory="db",
)

llm = GPT4All(
    model="./ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=1000, backend="gptj", verbose=False
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,
)

res = qa(
    f"""
    I want you to answer B's answer in a word.
    Extract it from the text.
"""
)
print(res["result"])
