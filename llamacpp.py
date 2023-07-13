from llama_cpp import Llama

llm = Llama(model_path="./wizardLM-7B.ggmlv3.q8_0.bin")
inputdata = ""

while True:
    inputdata = input("Input : ")
    output = llm(
        "Q: 'Person A: "
        + inputdata
        + "\n Person B: No, why?' In above chat history, Is Person B answered the Person A's question?"
        + " A: ",
        max_tokens=64,
        stop=["Q:", "\n"],
        echo=True,
    )
    print(output)
