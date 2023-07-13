from llama_cpp import Llama

llm = Llama(model_path="./wizardLM-7B.ggmlv3.q8_0.bin")
inputdata = ""
json_fields = ["zip_code", "work_tech", "company"]
questions = [
    {
        "main": "What's your zip code",
        "addin_prompt": "",
    },
    {
        "main": "Do you work in tech",
        "addin_prompt": "You should answer 'True' or 'False'",
    },
    {
        "main": "Which company did you last work for",
        "addin_prompt": "",
    },
]
i = 0

while True:
    i = 0
    while i < 3:
        print(questions[i]["main"] + "?")
        answer = input("Answer : ")
        output = llm(
            "Q: 'Jackson: "
            + questions[i]["main"]
            + "\n Neil: "
            + answer
            + "' \n "
            + "What is Neil's "
            + json_fields[i]
            + " in above chat exactly in a word? If you can't find the Niel's correct answer correspond to the question in a word, return 'none'. A: ",
            max_tokens=64,
            stop=["Q:", "\n"],
            echo=True,
        )
        print(output)
        i = i + 1

    # What is the zip code in Person B's answer in a word?
