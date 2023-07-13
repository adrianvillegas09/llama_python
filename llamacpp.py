from llama_cpp import Llama
from langchain import PromptTemplate, LLMChain


llm = Llama(model_path="./wizardLM-7B.ggmlv3.q8_0.bin")
template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)

inputdata = ""
json_fields = ["zip_code", "work_tech", "company"]

while True:
    inputdata = input("Input : ")
    answer = input("B's Answer : ")
    output = llm(
        "Q: 'Person A: "
        + "What is your zip code?"
        + "\n Person B: "
        + answer
        + "' \n In above chat history, "
        + " A: ",
        max_tokens=64,
        stop=["Q:", "\n"],
        echo=True,
    )
    print(output)

    # What is the zip code in Person B's answer in a word?
