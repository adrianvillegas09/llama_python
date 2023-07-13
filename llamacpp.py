from llama_cpp import Llama

llm = Llama(model_path="./wizardLM-7B.ggmlv3.q8_0.bin")
inputdata = ""

while True:
    inputdata = input()
    output = llm(
        "Q: 'A: What is your zip code? B:No, why?'. Extract the info from above chat history."
        + inputdata
        + " A: ",
        max_tokens=64,
        stop=["Q:", "\n"],
        echo=True,
    )
    print(output)
