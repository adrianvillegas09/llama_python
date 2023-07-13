from llama_cpp import Llama

llm = Llama(model_path="./wizardLM-7B.ggmlv3.q8_0.bin")
inputdata = ""

while True:
    inputdata = input("Input : ")
    output = llm(
        "Q: " + inputdata + " A: No, why?",
        max_tokens=64,
        stop=["Q:", "\n"],
        echo=True,
    )
    print(output)
