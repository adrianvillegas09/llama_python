from llama_cpp import Llama

llm = Llama(model_path="./wizardLM-7B.ggmlv3.q8_0.bin")
inputdata = ""

while True:
    inputdata = input("Input : ")
    output = llm(
        "Q: 'A: "
        + inputdata
        + "B: No, why?' Can you answer the B's answer in a word?"
        + " A: ",
        max_tokens=64,
        stop=["Q:", "\n"],
        echo=True,
    )
    print(output)
