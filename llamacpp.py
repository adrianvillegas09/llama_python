from llama_cpp import Llama

llm = Llama(model_path="./wizardLM-7B.ggmlv3.q8_0.bin")
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
        + inputdata
        + " A: ",
        max_tokens=64,
        stop=["Q:", "\n"],
        echo=True,
    )
    print(output)

    # What is the zip code in Person B's answer in a word?
